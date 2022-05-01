import os
import gzip
import torch
import shutil
import zipfile
import requests

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torchvision.transforms as transforms


class MNISTDataset(Dataset):
    def __init__(self, K, Q=5):

        self.mnist_dir = './data/MNIST/'

        self.K = K
        self.Q = Q

        self.char_path = [folder for folder, folders, _ in os.walk(self.mnist_dir) if not folders]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.char_path)

    def __getitem__(self, idx):

        img = []
        for img_ in os.listdir(self.char_path[idx]):  # fixme: sort?
            if True:
                if 'png' in img_:
                    img.append(self.transform(Image.open(self.char_path[idx] + '/' + img_, mode='r').convert('L')))
            else:
                if 'pt' in img_:
                    img.append(torch.load(self.char_path[idx] + '/' + img_))

        img = torch.cat(img)
        idx_vec = idx * torch.ones_like(torch.empty(890), dtype=int)

        return img[:self.K], idx_vec[:self.K], img[self.K:self.K + self.Q], idx_vec[self.K:self.K + self.Q]


class EmnistDataset(Dataset):
    def __init__(self, K, Q=5):
        try:
            # -- create directory
            s_dir = os.getcwd()
            self.emnist_dir = s_dir + '/data/emnist/'
            file_name = 'gzip'
            os.makedirs(self.emnist_dir)

            # -- download
            emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/'
            self.download(emnist_url + file_name + '.zip', self.emnist_dir + file_name + '.zip')

            # -- unzip
            with zipfile.ZipFile(self.emnist_dir + file_name + '.zip', 'r') as zip_file:
                zip_file.extractall(self.emnist_dir)
            os.remove(self.emnist_dir + file_name + '.zip')

            balanced_path = [f for f in [fs for _, _, fs in os.walk(self.emnist_dir + file_name)][0] if 'balanced' in f]
            for file in balanced_path:
                with gzip.open(self.emnist_dir + 'gzip/' + file, 'rb') as f_in:
                    try:
                        f_in.read(1)
                        with open(self.emnist_dir + file[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    except OSError:
                        pass
            shutil.rmtree(self.emnist_dir + file_name)

            # -- write images
            self.write_to_file()

            remove_path = [files for _, folders, files in os.walk(self.emnist_dir) if folders][0]
            for path in remove_path:
                os.unlink(self.emnist_dir + path)

        except FileExistsError:
            pass

        self.K = K
        self.Q = Q

        # --
        self.char_path = [folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders]
        self.transform = transforms.Compose([transforms.ToTensor()])  # fixme

    @staticmethod
    def download(url, filename):
        res = requests.get(url, stream=False)
        with open(filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def write_to_file(self):
        n_class = 47

        # -- read images and labels
        with open(self.emnist_dir + 'emnist-balanced-test-images-idx3-ubyte', 'rb') as f:
            f.read(3)
            image_count = int.from_bytes(f.read(4), 'big')
            height = int.from_bytes(f.read(4), 'big')
            width = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape((image_count, height, width))  # fixme: rotate?

        with open(self.emnist_dir + 'emnist-balanced-test-labels-idx1-ubyte', 'rb') as f:
            f.read(3)
            label_count = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        assert (image_count == label_count)

        # -- write images
        for i in range(n_class):
            os.mkdir(self.emnist_dir + f'character{i + 1:02d}')

        char_path = sorted([folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders])

        label_counter = np.ones(n_class, dtype=int)
        for i in range(label_count):
            im = Image.fromarray(images[i])
            im.save(char_path[labels[i]] + f'/{labels[i] + 1:02d}_{label_counter[labels[i]]:04d}.png')

            label_counter[labels[i]] += 1

    def __len__(self):
        return len(self.char_path)

    def __getitem__(self, idx):

        img = []
        for img_ in os.listdir(self.char_path[idx]):  # fixme: sort?
            if True:
                if 'png' in img_:
                    img.append(self.transform(Image.open(self.char_path[idx] + '/' + img_, mode='r').convert('L')))
            else:
                if 'pt' in img_:
                    img.append(torch.load(self.char_path[idx] + '/' + img_))

        img = torch.cat(img)
        idx_vec = idx * torch.ones_like(torch.empty(400), dtype=int)

        return img[:self.K], idx_vec[:self.K], img[self.K:self.K + self.Q], idx_vec[self.K:self.K + self.Q]


class OmniglotDataset(Dataset):
    def __init__(self, K, Q=5, dim=84):
        try:
            # -- create directory
            s_dir = os.getcwd()
            omniglot_dir = s_dir + '/data/omniglot/'
            file_name = 'images_background'
            os.makedirs(omniglot_dir)

            # -- download
            omniglot_url = 'https://github.com/brendenlake/omniglot/raw/master/python'
            self.download(omniglot_url + '/' + file_name + '.zip', omniglot_dir + file_name + '.zip')

            # -- unzip
            with zipfile.ZipFile(omniglot_dir + file_name + '.zip', 'r') as zip_file:
                zip_file.extractall(omniglot_dir)
        except FileExistsError:
            pass

        self.path = omniglot_dir + file_name
        self.K = K
        self.Q = Q

        # --
        self.char_path = [folder for folder, folders, _ in os.walk(self.path) if not folders]
        self.transform = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])

    @staticmethod
    def download(url, filename):
        res = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def __len__(self):
        return len(self.char_path)

    def __getitem__(self, idx):

        img = []
        for img_ in os.listdir(self.char_path[idx]): 
            if 'png' in img_:
                img.append(self.transform(Image.open(self.char_path[idx] + '/' + img_, mode='r').convert('L')))

        img = torch.cat(img)
        idx_vec = idx * torch.ones_like(torch.empty(20), dtype=int)

        return img[:self.K], idx_vec[:self.K], img[self.K:self.K + self.Q], idx_vec[self.K:self.K + self.Q]


class DataProcess:
    def __init__(self, M, K, Q, database, dim, device='cpu', iid=True):
        self.M = M
        self.K = K
        self.Q = Q
        self.database = database
        self.device = device
        self.iid = iid
        self.dim = dim
        self.dim_ = 784

    def __call__(self, data):

        x_trn, y_trn, x_qry, y_qry = data

        x_trn = torch.reshape(x_trn, (self.M * self.K, self.dim_)).to(self.device)
        y_trn = torch.reshape(y_trn, (self.M * self.K, 1)).to(self.device)
        x_qry = torch.reshape(x_qry, (self.M * self.Q, self.dim_)).to(self.device)
        y_qry = torch.reshape(y_qry, (self.M * self.Q, 1)).to(self.device)

        if self.iid:
            perm = np.random.choice(range(self.M * self.K), self.M * self.K, False)

            x_trn = x_trn[perm]
            y_trn = y_trn[perm]

        return x_trn, y_trn, x_qry, y_qry

