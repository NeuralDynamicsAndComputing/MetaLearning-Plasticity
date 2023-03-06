import os
import gzip
import torch
import shutil
import zipfile
import requests

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np


class EmnistDataset(Dataset):
    """
        EMNIST Dataset class.

    Constructs training and query sets for meta-training. Note that rather
    than a single image and the corresponding label, each data point
    represents samples from a class of images, containing training and query
    data from that category.
    """
    def __init__(self, K, Q, dim):
        """
            Initialize the EmnistDataset class.

        The method first downloads and preprocesses the EMNIST dataset, creating
        directories and files necessary for later use. It then sets the values for
        K, Q, and dim, which define the number of training data, the number of queries,
        and the dimensions of the images to be loaded, respectively. It also sets the
        path to the images and defines the transformation to be applied to the images.

        :param K: (int) integer value representing the number of training data per class,
        :param Q: (int) integer value representing the number of query data per class,
        :param dim: (int) integer value representing the dimension size of the images.
        """
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

        self.char_path = [folder for folder, folders, _ in os.walk(self.emnist_dir) if not folders]
        self.transform = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])

    @staticmethod
    def download(url, filename):
        """
            A static method to download a file from a URL and save it to a local file.

        :param url: (str) A string representing the URL from which to download the file,
        :param filename: (str) A string representing the name of the local file to save
            the downloaded data to.
        :return: None
        """
        res = requests.get(url, stream=False)
        with open(filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def write_to_file(self):
        """
            Write EMNIST images to files.

        The function reads the EMNIST test images and labels from binary files and
        writes them to files. Each image is saved to a file under a directory
        corresponding to its label.

        :return: None.
        """
        n_class = 47

        # -- read images
        with open(self.emnist_dir + 'emnist-balanced-test-images-idx3-ubyte', 'rb') as f:
            f.read(3)
            image_count = int.from_bytes(f.read(4), 'big')
            height = int.from_bytes(f.read(4), 'big')
            width = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape((image_count, height, width))

        # -- read labels
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
        """
            Get the length of the dataset.

        :return: int: the length of the dataset, i.e., the number of classes in the
            dataset
        """
        return len(self.char_path)

    def __getitem__(self, idx):
        """
            Return a tuple of tensors containing training and query images and
            corresponding labels for a given index.

        The images are loaded from the character folder at the given index. Each
        image is converted to grayscale and resized to the specified dimension
        using `torchvision.transforms.Resize` and `torchvision.transforms.ToTensor`.
        The returned tuples contain tensors of K and Q images, where K is the training
        data size per class and Q is the query data size per class. Both K and Q are
        specified at initialization. The indices corresponding to the images are
        also returned in tensors of size K and Q, respectively.

        :param idx: (int) Index of the character folder from which images are to be retrieved.
        :return: tuple: A tuple (img_K, idx_vec_K, img_Q, idx_vec_Q) containing the following tensors:
            - img_K (torch.Tensor): A tensor of K images from the character folder at idx
                as training data.
            - idx_vec_K (torch.Tensor): A tensor of K indices corresponding to the images
                in img_K.
            - img_Q (torch.Tensor): A tensor of Q images from the character folder at idx
                as query data.
            - idx_vec_Q (torch.Tensor): A tensor of Q indices corresponding to the images
                in img_Q.
        """
        img = []
        for img_ in os.listdir(self.char_path[idx]):
            img.append(self.transform(Image.open(self.char_path[idx] + '/' + img_, mode='r').convert('L')))

        img = torch.cat(img)
        idx_vec = idx * torch.ones_like(torch.empty(400), dtype=int)

        return img[:self.K], idx_vec[:self.K], img[self.K:self.K + self.Q], idx_vec[self.K:self.K + self.Q]


class DataProcess:
    """
        Meta-training data processor class.

    The function is designed to process meta-training data, specifically
    training and query data sets. The function performs several operations,
    including:
    1) Flattening images and merging image category and image index dimensions,
    2) Transferring the processed data to the specified processing device,
        which could either be 'cpu' or 'cuda',
    3) Shuffling the order of data points in the training set to avoid any
        potential biases during model training.
    """
    def __init__(self, K, Q, dim, device='cpu', iid=True):
        """
            Initialize the DataProcess object.

        :param K: (int) training data set size per class,
        :param Q: (int) query data set size per class,
        :param dim: (int) image dimension,
        :param device: (str) The processing device to use. Default is 'cpu',
        :param iid: (bool) shuffling flag. Default is True.
        """
        self.K = K
        self.Q = Q
        self.device = device
        self.iid = iid
        self.dim = dim

    def __call__(self, data, M):
        """
            Processing meta-training data.

        :param data: (tuple) A tuple of training and query data and the
            corresponding indices.
        :param M: (int) The number of classes.
        :return: tuple: A tuple of processed training and query data and
            the corresponding indices.
        """

        # -- load data
        x_trn, y_trn, x_qry, y_qry = data

        # -- reshape
        x_trn = torch.reshape(x_trn, (M * self.K, self.dim ** 2)).to(self.device)
        y_trn = torch.reshape(y_trn, (M * self.K, 1)).to(self.device)
        x_qry = torch.reshape(x_qry, (M * self.Q, self.dim ** 2)).to(self.device)
        y_qry = torch.reshape(y_qry, (M * self.Q, 1)).to(self.device)

        # -- shuffle
        if self.iid:
            perm = np.random.choice(range(M * self.K), M * self.K, False)

            x_trn = x_trn[perm]
            y_trn = y_trn[perm]

        return x_trn, y_trn, x_qry, y_qry
