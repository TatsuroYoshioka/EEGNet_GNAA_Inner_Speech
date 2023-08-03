import os

from torch.utils.data import DataLoader
import numpy as np
import torch

from seed_func import fix_seed
from dataset import load_inner_speech_dataset, InnerSpeechDataset
from model import EEGNet

#epoch
epochs=100

#subject
#sub_num_set=[3]
sub_num_set=[1,2,3,4,5,6,7,8,9]

#condition
condition_num_set = [1] #pattern2,3
#condition_num_set = [0,2,1] #pattern4

#band-pass filter
filtering = True
Low_cut = 20
High_cut = 40

#model_name
model_name = ''

firstLoop = True
SEED = 42
fix_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for condition_num in condition_num_set:
    for sub_num in sub_num_set:
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

        dataset_dict = {'train': [], 'valid': []}
        dataloader_dict = {'train': [], 'valid': []}

        phases = ['train', 'valid']
        X = np.array([i for i in range(len(data))])
        for i in range(len(X)):
            #np.random.shuffle(ids)
            # train : valid = 85 : 15
            ids_train = X[:int(len(X)*17/20//1)]
            ids_valid = X[int(len(X)*17/20//1):]
            ids_dict = {'train': ids_train, 'valid': ids_valid}

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

        model = EEGNet()
        model = model.to(device)
        if firstLoop == False:
            model.load_state_dict(torch.load(model_name))

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
            for _, (inputs, labels) in enumerate(dataloader_dict['train'][0]):
                labels = labels.to('cpu')
                n_train += labels.size()[0]
                optimizer.zero_grad()
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                train_loss = criterion(outputs, labels)
                train_loss_epoch += train_loss.item()
                for i in range(len(labels)):
                    if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
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
                for _, (inputs, labels) in enumerate(dataloader_dict['valid'][0]):
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
                torch.save(model.state_dict(), model_name)

            valid_acc_list.append(valid_acc/n_valid)
            valid_loss_list.append(valid_loss_epoch/n_valid)

            print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
                epoch + 1,
                train_loss_epoch/n_train,
                train_acc/n_train,
                valid_loss_epoch/n_valid,
                valid_acc/n_valid
            ))

        if firstLoop == True:
            firstLoop == False
        