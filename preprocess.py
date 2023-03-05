import os
import torch
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from kymatio.torch import Scattering2D


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- embedding params
        self.cn1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.cn2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.cn3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)

        # -- non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.cn1(x))
        x = self.relu(self.cn2(x))
        x = self.relu(self.cn3(x))

        return x


def data_process():

    # -- init
    model_conv = True
    s_dir = os.getcwd()
    emnist_dir = s_dir + '/data/emnist/'
    char_path = [folder for folder, folders, _ in os.walk(emnist_dir) if not folders]
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- init model
    if model_conv:
        model = MyModel().to(device)
        path_pretrained = './data/models/model_stat.pth'
        old_model = torch.load(path_pretrained, map_location=device)
        for old_key in old_model:
            try:
                dict(model.named_parameters())[old_key].data = old_model[old_key]
            except:
                pass
    else:
        model = Scattering2D(J=3, L=4, shape=(28, 28), max_order=2)

    # -- process
    for idx in range(len(char_path)):
        files = [files for _, _, files in os.walk(char_path[idx])][0]
        for img in files:
            if 'png' in img:
                data = transform(Image.open(char_path[idx] + '/' + img, mode='r').convert('L'))
                if model_conv:
                    data_processed = model(data.unsqueeze(0).to(device))
                else:
                    data_processed = model(data)
                data_processed = data_processed.view(data_processed.size(0), -1)

                # -- store
                torch.save(data_processed.data, char_path[idx] + '/' + img[:-4] + '.pt')

data_process()
