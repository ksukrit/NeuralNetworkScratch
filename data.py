import numpy as np
import os

class Dataset:
    def __init__(self,path) -> None:
        dir_files = os.listdir(path)
        self.data = []
        self.labels = []
        for file in dir_files:
            if file.endswith(".npy"):
                self.data.append(np.load(path + file))
                self.labels.append(file.split("_")[0])
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.data = self.data.reshape(self.data.shape[0],-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __iter__(self):
        return iter(zip(self.data, self.labels))
    
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_indices = self.indices[self.index:self.index+self.batch_size]
        batch = self.dataset[batch_indices]
        self.index += self.batch_size
        return batch