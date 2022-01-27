
import torch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision import transforms, datasets

from torch import nn, optim
from torch.utils.data import DataLoader


def log(data, filename):

    with open(filename, 'a') as f:
        np.savetxt(f, np.array(data), newline=' ', fmt='%0.6f')
        f.writelines('\n')
        

def plot_meta(filename, title, y_lim, K, res_dir, data_type='.png'):

    y = np.loadtxt(res_dir + '/' + filename)
    y = np.nan_to_num(y)

    plt.plot(np.array(range(len(y))), y)
    plt.title(title + ', $K={}$'.format(K))
    plt.ylim(y_lim)
    plt.savefig(res_dir + '/' + title + '_K' + str(K) + data_type, bbox_inches='tight')
    plt.close()


def plot_adpt(filename, title, y_lim, K, res_dir, data_type='.png'):
    y = np.loadtxt(res_dir + '/' + filename)
    y = np.nan_to_num(y)
    for idx in range(0, y.shape[0], 100):
        plt.plot(np.array(range(y.shape[1])), y[idx])
    plt.legend(range(0, y.shape[0], 100))
    plt.title(title + ', $K={}$'.format(K))
    plt.ylim(y_lim)
    plt.savefig(res_dir + '/' + title + '_K' + str(K) + data_type, bbox_inches='tight')
    plt.close()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        dim_out = 10

        # -- embedding params
        self.cn1 = nn.Conv2d(1, 16, 7)
        self.cn2 = nn.Conv2d(16, 32, 6)
        self.cn3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # -- prediction params
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, dim_out)

        # -- non-linearity
        self.relu = nn.ReLU()
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

    def forward(self, x):

        x = self.relu(self.bn1(self.cn1(x)))
        x = self.pool(x)
        x = self.relu(self.cn2(x))
        x = self.relu(self.cn3(x))

        x = x.view(x.size(0), -1)

        x = self.sopl(self.fc1(x))
        x = self.sopl(self.fc2(x))

        return self.fc3(x)


class Train:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- data
        dim = 28
        batch_size = 100
        my_transforms = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])

        trainset = datasets.FashionMNIST('./data/fashionMNIST/', train=True, download=True, transform=my_transforms)
        validset = datasets.FashionMNIST('./data/fashionMNIST/', train=False, download=True, transform=my_transforms)

        self.TrainDataset = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        self.ValidDataset = DataLoader(dataset=validset, batch_size=len(validset), shuffle=True)

        self.N_train = len(trainset)
        self.N_valid = len(validset)

        # -- model
        self.model = MyModel().to(self.device)

        # -- train
        self.epochs = 3000
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    @staticmethod
    def accuracy(logits, label):

        pred = F.softmax(logits, dim=1).argmax(dim=1)

        return torch.eq(pred, label).sum().item()

    def train_epoch(self):

        self.model.train()
        train_loss = 0
        for batch_idx, data_batch in enumerate(self.TrainDataset):

            data_batch[0], data_batch[1] = data_batch[0].to(self.device), data_batch[1].to(self.device)

            # -- predict
            predict = self.model(data_batch[0])

            # -- loss
            loss = self.loss(predict, data_batch[1])

            # -- optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss/(batch_idx+1)

    def valid_epoch(self):

        with torch.no_grad():
            self.model.eval()
            for data_batch in self.ValidDataset:

                data_batch[0], data_batch[1] = data_batch[0].to(self.device), data_batch[1].to(self.device)

                # -- predict
                predict = self.model(data_batch[0])

                # -- loss
                loss = self.loss(predict, data_batch[1])

                # -- accuracy
                acc = self.accuracy(predict, data_batch[1]) / len(self.ValidDataset.dataset)
        
        return loss.item(), acc 

    def __call__(self):

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            valid_loss, acc = self.valid_epoch()

            print('Epoch {}: Training loss = {:.5f}, Validation loss = {:.5f}, Validation accuracy = {:.1f}%'.format(epoch, train_loss, valid_loss, acc*100.))

        torch.save(self.model.state_dict(), './model_stat.pth')


if __name__ == '__main__':
    my_train = Train()
    my_train()
