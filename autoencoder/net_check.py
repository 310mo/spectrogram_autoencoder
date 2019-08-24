import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import prepro

img_transform = transforms.Compose([
    transforms.ToTensor()
])

root = 'spect-16000'

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=4, padding=1, output_padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=4, padding=1, output_padding=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

test_dataset = prepro.CustomDataset(root, train=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

device = 'cpu'
net = AutoEncoder()
net.load_state_dict(torch.load('net.ckpt'))

net.eval()
with torch.no_grad():
    count = 0
    for image, label in test_loader:
        image = image.to(device, dtype=torch.float)
        np.save('input/input{}'.format(count), image)
        output = net(image)
        np.save('output/output{}'.format(count), output)
        count += 1
        break