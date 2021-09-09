import numpy as np
import torch
from keras.datasets import mnist

np.random.seed(42)


def load_data(n_train):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.reshape(X_train[:n_train,:,:], (n_train, 784)).T
    y_train = np.reshape(y_train[:n_train], (n_train, 1)).T

    return X_train, y_train


class Linear:
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        w = 1. / np.sqrt(input_size)
        self.weight = np.random.uniform(-w, w, (output_size, input_size))

    def __call__(self, x):
        return np.matmul(self.weight, x)


class model:
    def __init__(self):
        super(model, self).__init__()

        self.h_1 = Linear(784, 512)
        self.h_2 = Linear(512, 264)
        self.h_3 = Linear(264, 128)
        self.h_4 = Linear(128, 1)

    @property
    def get_layers(self):
        return {1: self.h_1, 2: self.h_2, 3: self.h_3, 4: self.h_4}

    def ReLU(self, x):
        return np.maximum(np.zeros(x.shape), x)

    def __call__(self, y0):

        y1 = self.h_1(y0)
        y2 = self.h_2(y1)
        y3 = self.h_3(y2)
        y4 = self.h_4(y3)

        return y0, y1, y2, y3, y4


class train(object):
    def __init__(self, X_train, y_train, batch_size, eta=1e-3):

        self.model = model()
        self.epochs = 100
        self.n_layers = len(self.model.get_layers)
        self.eta = eta
        self.batch_size = batch_size
        self.batch_n = -(-len(X_train.T)//batch_size)

        self.X_train = X_train
        self.y_train = y_train

    def del_W(self, y_lm1, e_l):
        """
            Weight update rule.
        :param y_lm1: inputs to the layer.
        :param e_l: error vector.
        :return: weight update
        """
        return - self.eta * np.matmul(e_l, y_lm1.T)

    def trainepoch(self, epoch):
        """
            Single epoch training.
        :param epoch: current epoch number.
        """

        train_loss = 0
        for idx in range(self.batch_n):

            # -- training data
            y0 = self.X_train[:, idx * self.batch_size:(idx + 1) * self.batch_size]/256
            y_target = self.y_train[:, idx * self.batch_size:(idx + 1) * self.batch_size]

            # -- predict
            y = self.model(y0)

            # -- compute error
            e = [y_target - y[-1]]
            for l in range(4, 1, -1):
                e.insert(0, np.matmul(self.model.get_layers[l].weight.T, e[0]))

            # -- weight update
            for l, key in enumerate(self.model.get_layers.keys()):
                self.model.get_layers[key].weight = self.model.get_layers[key].weight + self.del_W(y[l], e[l])

            # -- compute loss
            train_loss += 0.5 * np.matmul(e[-1], e[-1].T).item()

        # -- log
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / n_train))

    def __call__(self):
        """
            Model training.
        """

        for epoch in range(1,self.epochs+1):
            self.trainepoch(epoch)

n_train = 200
batch_size = 10

X_train, y_train = load_data(n_train)
my_train = train(X_train, y_train, batch_size)

my_train()
