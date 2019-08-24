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

learning_rate = 0.001

#16000Hzにしたスペクトログラムはspect-16000/以下に置く
root = 'spect-16000'
train_dataset = prepro.CustomDataset(root, train=True)
test_dataset = prepro.CustomDataset(root)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AutoEncoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

num_epochs = 1000

loss_list = []

for epoch in range(num_epochs):

    net.train()
    for data in train_loader:
        img, _ = data
        img = img.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = net(img)
        #print(outputs.size())
        #print(img.size())
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()

        #logging
        loss_list.append(loss.item())

    print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

torch.save(net.state_dict(), 'net.ckpt')

