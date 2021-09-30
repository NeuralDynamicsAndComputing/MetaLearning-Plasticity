import os
import torch
import zipfile
import requests

from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
        list_alph = [[a_dir + '/' + j for j in os.listdir(self.path + '/' + a_dir)] for a_dir in os.listdir(self.path)]
        self.char_list = [item for sublist in list_alph for item in sublist]

        self.transform = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])

    @staticmethod
    def download(url, filename):
        res = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def __len__(self):
        return len(self.char_list)

    def __getitem__(self, idx):

        image = []
        for img in os.listdir(self.path + '/' + self.char_list[idx]):
            img_path = self.path + '/' + self.char_list[idx] + '/' + img
            image.append(self.transform(Image.open(img_path, mode='r').convert('L')))

        image = torch.cat(image)
        index_vec = idx*torch.ones_like(torch.empty(20))

        return image[:self.steps], index_vec[:self.steps], \
               image[self.steps:self.steps+5], index_vec[self.steps:self.steps+5]


tasks = 5
steps = 5

TrainDataset = DataLoader(dataset=OmniglotDataset(steps), batch_size=tasks, shuffle=True)

for idx, (img_trn, lbl_trn, img_tst, lbl_tst) in enumerate(TrainDataset):

    img_tst = torch.reshape(img_tst, (tasks * 5,  84, 84))
    lbl_tst = torch.reshape(lbl_tst, (1, tasks * 5))

    for img, label in zip(torch.reshape(img_trn, (tasks * steps, 84, 84)), torch.reshape(lbl_trn, (tasks * steps, 1))):

        print(label)

    print(lbl_tst)

    quit()
