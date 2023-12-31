import os

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

from seed_func import fix_seed
from dataset import load_inner_speech_dataset, InnerSpeechDataset
from model import EEGNet
# from confusion_matrix import pp_matrix_from_data

#pramater
#stratified k-fold cross validation
k=10
#epoch
epochs=100
#subject
sub_num = 10
#condition
condition_num = 1
#band-pass filter
filtering = True
Low_cut = 20
High_cut = 40
#use pre-training model
pretrain = False
#pre-training model path
#model_dir = 
#gradient norm adversarial augentation
gnaa = False
epsilon = 0.001 #amount of perturbation

#confusion matrix save directory
#save_dir = 

SEED = 42
fix_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create datasets and dataloaders (k-folds)
data_paths = []
events_paths = []
for ses_num in range(1, 4):
    sub_str = str(sub_num).zfill(2)
    ses_str = str(ses_num).zfill(2)
    data_path = '/root/dataset/derivatives/sub-{}/ses-{}/sub-{}_ses-{}_eeg-epo.fif'.format(
        sub_str, ses_str, sub_str, ses_str
    )
    data_paths.append(data_path)
    events_path = '/root/dataset/derivatives/sub-{}/ses-{}/sub-{}_ses-{}_events.dat'.format(
        sub_str, ses_str, sub_str, ses_str
    )
    events_paths.append(events_path)

data, labels = load_inner_speech_dataset(data_paths, events_paths, filtering, Low_cut, High_cut, condition_num)

dataset_dict = {'train': [], 'valid': [], 'test': []}
dataloader_dict = {'train': [], 'valid': [], 'test': []}

phases = ['train', 'valid', 'test']
X = np.array([i for i in range(len(data))])
y = np.argmax(labels, axis=1)
kf =  StratifiedKFold(n_splits = k, shuffle = True, random_state=SEED)
for ids_train_valid, ids_test in kf.split(X, y):
    np.random.shuffle(ids_train_valid)
    # train : valid = 85 : 15
    ids_train = ids_train_valid[:int(len(ids_train_valid)*17/20//1)]
    ids_valid = ids_train_valid[int(len(ids_train_valid)*17/20//1):]
    ids_dict = {'train': ids_train, 'valid': ids_valid, 'test': ids_test}

    for phase in phases:
        dataset = InnerSpeechDataset(data[ids_dict[phase]], labels[ids_dict[phase]])
        dataset_dict[phase].append(dataset)

        if phase == 'train':
            batch_size = 8
        else:
            batch_size = 1
        dataloader_dict[phase].append(DataLoader(dataset, batch_size=batch_size))

# experiment
criterion = torch.nn.CrossEntropyLoss()

test_acc_sum = 0
predict_label=[]
actual_label=[]
os.makedirs('result', exist_ok=True)
text_name = 'result/output_sub{}.txt'.format(str(sub_num).zfill(2))
for fold_i in range(k):
    model = EEGNet()
    model = model.to(device)
    if pretrain:
        model.load_state_dict(torch.load(model_dir))
    model_path = 'model.pth'

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    max_valid_acc = 0
    for epoch in range(epochs):
        # train
        model.train()
        train_acc = 0
        train_loss_epoch = 0
        n_train = 0
        for _, (inputs, labels) in enumerate(dataloader_dict['train'][fold_i]):
            labels = labels.to('cpu')
            n_train += labels.size()[0]
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            inputs.requires_grad = True
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss_epoch += train_loss.item()
            for i in range(len(labels)):
                if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                    train_acc += 1
            train_loss.backward()
            optimizer.step()

            if gnaa == True:
                grad = inputs.grad
                norm_grad = torch.norm(grad, p='fro')
                inputs_adv = inputs + epsilon*(grad/norm_grad)

                labels = labels.to('cpu')
                n_train += labels.size()[0]
                optimizer.zero_grad()
                inputs_adv = inputs_adv.type(torch.FloatTensor).to(device)
                labels = labels.to(device)

                outputs_adv = model(inputs_adv)
                train_loss = criterion(outputs_adv, labels)
                train_loss_epoch += train_loss.item()
                for i in range(len(labels)):
                    if torch.argmax(outputs_adv[i]) == torch.argmax(labels[i]):
                        train_acc += 1
                train_loss.backward()
                optimizer.step()

        train_acc_list.append(train_acc/n_train)
        train_loss_list.append(train_loss_epoch/n_train)

        # valid
        model.eval()
        valid_acc = 0
        valid_loss_epoch = 0
        n_valid = 0
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(dataloader_dict['valid'][fold_i]):
                labels = labels.to('cpu')
                n_valid += labels.size()[0]
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                valid_loss = criterion(outputs, labels)
                valid_loss_epoch += valid_loss
                for i in range(len(labels)):
                    if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                        valid_acc += 1
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(model.state_dict(), model_path)

        valid_acc_list.append(valid_acc/n_valid)
        valid_loss_list.append(valid_loss_epoch/n_valid)

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch + 1,
            train_loss_epoch/n_train,
            train_acc/n_train,
            valid_loss_epoch/n_valid,
            valid_acc/n_valid
        ))

    # test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_acc = 0
    n_test = 0
    predict = 0
    actual = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader_dict['test'][fold_i]):
            labels = labels.to('cpu')
            n_test += labels.size()[0]
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            for i in range(labels.shape[0]):
                if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                    test_acc += 1
                predict = torch.argmax(outputs[i]).to('cpu')
                actual = torch.argmax(labels[i]).to('cpu')
                predict_label.append(predict.item())
                actual_label.append(actual.item())

    test_acc_sum += test_acc/n_test
    with open(text_name, 'a') as f:
        f.write('test: ' + str(test_acc/n_test) + '\n')

    # plot training and validation accuracy and loss
    plt.plot(range(epochs), train_acc_list, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(range(epochs), valid_acc_list, color = "green", linestyle = "solid", label= 'valid acc')
    plt.title('training and Validation accuracy')
    plt.legend()
    plt.savefig('result/acc_fold{}.png'.format(fold_i))
    plt.close()

    plt.plot(range(epochs), train_loss_list, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(range(epochs), valid_loss_list, color = "orange", linestyle = "solid" , label= 'valid loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.savefig('result/loss_fold{}.png'.format(fold_i))
    plt.close()

# plot confusion matrix
# pp_matrix_from_data(actual_label, predict_label, save_dir=save_dir)

# calculate average test accuracy of k-folds
with open(text_name, 'a') as f:
    f.write('test_average: ' + str(test_acc_sum/k) + '\n')