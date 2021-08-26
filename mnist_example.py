import numpy as np

from keras.datasets import mnist

def load_data(n_train):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.reshape(X_train[:n_train,:,:], (n_train, 784))
    y_train = np.reshape(y_train[:n_train], (n_train, 1))

    return X_train, y_train

class Linear:
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        w = 0.00001#1. / np.sqrt(input_size)

        self.weight = np.random.uniform(-w, w, (input_size, output_size))

    def __call__(self, x):

        return np.dot(x, self.weight)

class model:
    def __init__(self):
        super(model, self).__init__()

        dim_size = 1500
        self.h_1 = Linear(784, dim_size)
        self.h_2 = Linear(dim_size, dim_size)
        self.h_3 = Linear(dim_size, dim_size)
        self.h_4 = Linear(dim_size, 1)

    @property
    def get_layers(self):
        return {1: self.h_1, 2: self.h_2, 3: self.h_3, 4: self.h_4}

    def ReLU(self, x):
        return x * (x > 0.)

    def __call__(self, y0):

        y1 = self.ReLU(self.h_1(y0))
        y2 = self.ReLU(self.h_2(y1))
        y3 = self.ReLU(self.h_3(y2))
        y4 = self.ReLU(self.h_4(y3))

        return y0, y1, y2, y3, y4

class train(object):
    def __init__(self, X_train, y_train):

        self.model = model()
        self.epochs = 10
        self.n_layers = len(self.model.get_layers)
        self.eta = 1e-2
        self.batch_size = 20
        self.batch_n = 100

        self.X_train = X_train
        self.y_train = y_train

    def trainepoch(self, epoch):

        train_loss = 0
        for idx in range(self.batch_n):

            y0 = self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size, :]
            y_target = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size]

            y = self.model(y0)

            e = []
            e.insert(0, y_target - y[-1])
            for i in range(4, 1, -1):
                e.insert(0, np.multiply(np.dot(e[0], np.transpose(self.model.get_layers[i].weight)), np.heaviside(y[i-1], 0.5)))

            for i in range(4):
                self.model.get_layers[i+1].weight -= self.eta * np.dot(np.transpose(y[i]), e[i])

            train_loss += 0.5 * np.dot(np.transpose(e[-1]), e[-1]).item()

        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / n_train))


    def __call__(self):

        for epoch in range(self.epochs):
            self.trainepoch(epoch)

n_train = 2000
X_train, y_train = load_data(n_train)
my_train = train(X_train, y_train)

my_train()
