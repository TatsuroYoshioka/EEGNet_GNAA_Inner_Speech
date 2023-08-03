from torch.utils.data import Dataset
import mne
import numpy as np


def load_inner_speech_dataset(data_paths, events_paths, filtering, Low_cut, High_cut, condition_num):
    data = []
    labels = []

    tmin = 1.0 
    tmax = 3.0 

    ses_events = dict()
    ses_data = dict()
    ses_num = 1

    for events_path in events_paths:
        ses_events[ses_num] = np.load(events_path, allow_pickle=True)
        ses_num += 1

    all_events = np.vstack((ses_events.get(1),ses_events.get(2),ses_events.get(3)))
    condition = all_events[:,2]

    ses_num = 1
    for data_path in data_paths:
        epochs = mne.read_epochs(data_path)
        epochs.crop(tmin=tmin, tmax=tmax)
        if filtering:
            epochs.filter(Low_cut, High_cut)
        ses_data[ses_num] = epochs._data
        ses_num += 1

    all_data = np.vstack((ses_data.get(1),ses_data.get(2),ses_data.get(3)))
    condition_events = all_events[condition == condition_num]
    condition_data = all_data[condition == condition_num]
    class_label = condition_events[:,1]
    for data_num in range(len(condition_data)):
        data.append(condition_data[data_num])
        if class_label[data_num] == 0:
            label = np.array([1, 0, 0, 0]).astype(float)
        if class_label[data_num] == 1:
            label = np.array([0, 1, 0, 0]).astype(float)
        if class_label[data_num] == 2:
            label = np.array([0, 0, 1, 0]).astype(float)
        if class_label[data_num] == 3:
            label = np.array([0, 0, 0, 1]).astype(float)
        labels.append(label)
    
    return np.array(data), np.array(labels)


class InnerSpeechDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)