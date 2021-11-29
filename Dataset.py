import os
import torch
import zipfile
import requests

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torchvision.transforms as transforms

n = 84
nxn = n * n

class OmniglotDataset(Dataset):
    def __init__(self, K, Q=5):
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
        self.transform = transforms.Compose([transforms.Resize((n, n)), transforms.ToTensor()])

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
            img.append(self.transform(Image.open(self.char_path[idx] + '/' + img_, mode='r').convert('L')))

        img = torch.cat(img)
        idx_vec = idx * torch.ones_like(torch.empty(20), dtype=int)

        return img[:self.K], idx_vec[:self.K], img[self.K:self.K + self.Q], idx_vec[self.K:self.K + self.Q]


def process_data(data, M=5, K=5, Q=5, iid=True):

    img_trn, lbl_trn, img_tst, lbl_tst = data

    img_tst = torch.reshape(img_tst, (M * Q,  n, n))
    lbl_tst = torch.reshape(lbl_tst, (M * Q, 1))
    img_trn = torch.reshape(img_trn, (M * K, n, n))
    lbl_trn = torch.reshape(lbl_trn, (M * K, 1))

    if iid:
        perm = np.random.choice(range(M * K), M * K, False)

        img_trn = img_trn[perm]
        lbl_trn = lbl_trn[perm]

    return img_trn, lbl_trn, img_tst, lbl_tst

