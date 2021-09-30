import os
import torch
import zipfile
import requests

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OmniglotDataset(Dataset):
    def __init__(self, ):
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

        return torch.cat(image), idx*torch.ones_like(torch.empty(20))
