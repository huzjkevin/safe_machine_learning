import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import sklearn.datasets as datasets
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Old method, will be updated or deprecated
def create_dataloader(data_path, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

#Old method, will be updated or deprecated
def create_test_dataloader(data_path, batch_size=1, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return testloader

def data_to_torch_dataset(data, target):
    data = torch.tensor(data, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.float).view(-1, 1)
    # target = torch.tensor(target, dtype=torch.long).view(-1, 1)
    return torch.utils.data.TensorDataset(data, target)

def boston_dataset():
    boston = datasets.load_boston()
    data = boston.data
    target = boston.target
    feature_dim = len(boston.feature_names)
    boston = data_to_torch_dataset(data, target)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0), std=(1.0))])
    return data_to_torch_dataset(data, target), feature_dim

class UCIDataset(Dataset):
    def __init__(self, data_path, transform=None, testing=False):
        data = np.genfromtxt(data_path, delimiter=",") # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        # #Keep the stat params for normalization.
        # self.stat = np.asarray([[np.mean(self.X), np.mean(self.Y)], # code for normalization new normalization
        #                         [np.std(self.X), np.std(self.Y)]])

        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))
        self.X = scaler.fit_transform(self.X)

        # code for normalization new normalization
        # if not testing:
        #     self.Y = scaler.fit_transform(self.Y.reshape(-1, 1))

        self.input_dim = len(self.X[0])

        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx])
        y = np.asarray(self.Y[idx])

        if self.transform:
            x = self.transform(x)

        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(y, dtype=torch.float).view(-1)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1)# return in form of tensor
        # return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1), self.stat # code for normalization new normalization


class WineDataset(Dataset):
    def __init__(self, data_path, transform=None, testing=False):
        data = np.genfromtxt(data_path, delimiter=",")[1:] # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))
        self.X = scaler.fit_transform(self.X)
        # self.Y = scaler.fit_transform(self.Y.reshape(-1, 1))
        self.input_dim = len(self.X[0])

        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx])
        y = np.asarray(self.Y[idx])

        if self.transform:
            x = self.transform(x.reshape(1, -1))

        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(y, dtype=torch.float).view(-1)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1) #return in form of tensor

if __name__ == "__main__":
    data_dir = os.path.join("./..", "data", "concrete")
    os.makedirs(data_dir, exist_ok=True)
    dataset_concrete = np.genfromtxt(os.path.join(data_dir, 'concrete.csv'), delimiter=',')[1:]
    X = dataset_concrete[:, :-1]
    Y = dataset_concrete[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2020, shuffle=True)
    pp_train = np.append(X_train, Y_train.reshape(-1, 1), axis=1)
    pp_test = np.append(X_test, Y_test.reshape(-1, 1), axis=1)

    np.savetxt(os.path.join(data_dir, "concrete_train.csv"), pp_train, delimiter=",")
    np.savetxt(os.path.join(data_dir, "concrete_test.csv"), pp_test, delimiter=",")

    concrete_train = np.genfromtxt(os.path.join(data_dir, 'concrete_train.csv'), delimiter=',')
    concrete_test = np.genfromtxt(os.path.join(data_dir, 'concrete_test.csv'), delimiter=',')
    print("finished")

#     def __getitem__(self, item):
#         x = np.asarray(self.X[item])
#         y = np.asarray(self.Y[item])
#
#         if self.transform is not None:
#             x = self.transform(x)
#
#         return torch.from_numpy(x), torch.from_numpy(y) #return in form of tensor
#
#
# if __name__ == '__main__':
#     #split_wine_dataset('./dataset/wine/winequality-white.csv')
#     dataset = WineDataset('../dataset/wine/test_winequality-white.csv', None)
#     data_loader = DataLoader(dataset,batch_size=1, shuffle=True, num_workers=0)
#     print((next(iter(data_loader))))
#     print(np.shape(next(iter(data_loader))[0]))
# >>>>>>> lhj/wineDataset
