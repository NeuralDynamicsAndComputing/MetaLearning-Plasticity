import os
import torch
import zipfile
import requests

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torchvision.transforms as transforms


class OmniglotDataset(Dataset):
    def __init__(self, steps):
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
        self.steps = steps

        # --
        self.char_path = [folder for folder, folders, _ in os.walk(self.path) if not folders]
        self.transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

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

        return img[:self.steps], idx_vec[:self.steps], img[self.steps:self.steps+5], idx_vec[self.steps:self.steps+5]


def Myload_data(data, tasks=5, steps=5, iid=True):

    img_trn, lbl_trn, img_tst, lbl_tst = data

    img_tst = torch.reshape(img_tst, (tasks * 5,  28, 28))
    lbl_tst = torch.reshape(lbl_tst, (1, tasks * 5))

    img_trn = torch.reshape(img_trn, (tasks * steps, 28, 28))
    lbl_trn = torch.reshape(lbl_trn, (tasks * steps, 1))

    if iid:
        perm = np.random.choice(range(tasks * steps), tasks * steps, False)

        img_trn = img_trn[perm]
        lbl_trn = lbl_trn[perm]

    return img_trn, lbl_trn, img_tst, lbl_tst


def main():
    tasks = 5
    steps = 5
    iid = True

    TrainDataset = DataLoader(dataset=OmniglotDataset(steps), batch_size=tasks, shuffle=True)

    for idx, data in enumerate(TrainDataset):

        img_trn, lbl_trn, img_tst, lbl_tst = Myload_data(data)

        for image, label in zip(img_trn, lbl_trn):

            print(img_trn[0])

            print(label)

        print(lbl_tst)

        quit()


if __name__ == '__main__':
    main()
