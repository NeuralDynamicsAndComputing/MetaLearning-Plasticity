import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
from torch import nn, optim

import torchvision


def log(data, filename):

    with open(filename, 'a') as f:
        np.savetxt(f, np.array(data), newline=' ', fmt='%0.6f')
        f.writelines('\n')


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        dim_out = 964

        # -- embedding params
        self.cn1 = nn.Conv2d(1, 16, 7)
        self.cn2 = nn.Conv2d(16, 32, 4)
        self.cn3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # -- prediction params
        self.fc1 = nn.Linear(256, 170)
        self.fc2 = nn.Linear(170, 50)
        self.fc3 = nn.Linear(50, dim_out)

        # -- non-linearity
        self.relu = nn.ReLU()
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

    def forward(self, x):

        y1 = self.pool(self.bn1(self.relu(self.cn1(x))))
        y2 = self.pool(self.bn2(self.relu(self.cn2(y1))))
        y3 = self.relu(self.bn3(self.cn3(y2)))


        y3 = y3.view(y3.size(0), -1)

        y5 = self.sopl(self.fc1(y3))
        y6 = self.sopl(self.fc2(y5))

        return self.fc3(y6)


class Train:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- data
        dim = 28
        batch_size = 400
        my_transforms = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])
        trainset = torchvision.datasets.Omniglot(root="./data/omniglot_train/", download=False, transform=my_transforms)
        validset = torchvision.datasets.Omniglot(root="./data/omniglot_train/", background=False, download=False,
                                                     transform=my_transforms)
        self.TrainDataset = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        self.ValidDataset = DataLoader(dataset=validset, batch_size=len(validset), shuffle=False)
        self.N_train = len(trainset)
        self.N_valid = len(validset)

        # -- model
        self.model = MyModel().to(self.device)

        # -- train
        self.epochs = 3000
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self):

        self.model.train()
        train_loss = 0
        for batch_idx, data_batch in enumerate(self.TrainDataset):
            # -- predict
            predict = self.model(data_batch[0].to(self.device))

            # -- loss
            loss = self.loss(predict, data_batch[1].to(self.device))

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
                # -- predict
                predict = self.model(data_batch[0].to(self.device))

                # -- loss
                loss = self.loss(predict, data_batch[1].to(self.device))

        return loss.item()

    def __call__(self):

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()

            print('Epoch {}: Training loss = {:.5f}, Validation loss = {:.5f}.'.format(epoch, train_loss, valid_loss))

        torch.save(self.model.state_dict(), './model_stat.pth')


if __name__ == '__main__':
    my_train = Train()
    my_train()



