import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

root = 'spect-16000'

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        #指定する場合は前処理クラスを受け取る
        self.transform = transform
        #画像とラベルの一覧を保持するリスト
        self.images = []
        self.labels = []
        #ルートフォルダーパス
        root = 'spect-16000'
        #訓練の場合と検証の場合でフォルダ分け
        #画像を読み込むファイルパスを取得
        if train==True:
            path = os.path.join(root, 'train')
        else:
            path = os.path.join(root, 'val')

        images = os.listdir(path)
        for image in images:
            self.images.append(os.path.join(path, image))
            if 'fujitou' in image:
                self.labels.append(0)
            elif 'tsuchiya' in image:
                self.labels.append(1)
            else:
                self.labels.append(2)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        with open(image, 'rb') as f:
            image = np.load(image)
            image = np.array([image])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)