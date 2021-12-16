import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
        self.cn1 = nn.Conv2d(1, 256, kernel_size=3, stride=2)
        self.cn2 = nn.Conv2d(256, 256, kernel_size=4, stride=1)
        self.cn3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn4 = nn.Conv2d(256, 256, kernel_size=4, stride=1)

        # -- prediction params
        self.fc1 = nn.Linear(256, 170)
        self.fc2 = nn.Linear(170, 50)
        self.fc3 = nn.Linear(50, dim_out)

        # -- non-linearity
        self.relu = nn.ReLU()
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

    def forward(self, x):

        y1 = self.relu(self.cn1(x))
        y2 = self.relu(self.cn2(y1))
        y3 = self.relu(self.cn3(y2))
        y4 = self.relu(self.cn4(y3))

        y4 = y4.view(y4.size(0), -1)

        y5 = self.sopl(self.fc1(y4))
        y6 = self.sopl(self.fc2(y5))

        return self.fc3(y6)


class Train:
    def __init__(self):

        # -- data
        dim = 28
        batch_size = 40
        my_transforms = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])
        dataset = torchvision.datasets.Omniglot(root="./data/omniglot_train/", download=False, transform=my_transforms)
        self.testset = torchvision.datasets.Omniglot(root="./data/omniglot_train/", background=False, download=False,
                                                     transform=my_transforms)
        self.TrainDataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        # -- model
        self.model = MyModel()

        # -- train
        self.epochs = 1000
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self, epoch):

        self.model.train()
        train_loss = 0
        for batch_idx, data_batch in enumerate(self.TrainDataset):
            # -- predict
            predict = self.model(data_batch[0])

            # -- loss
            loss = self.loss(predict, data_batch[1])

            # -- optimize
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            print(batch_idx, loss.item())

        print(epoch, train_loss)

    def __call__(self):

        for epoch in range(self.epochs):
            self.train_epoch(epoch)


my_train = Train()
my_train()

