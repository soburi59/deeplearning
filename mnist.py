import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

# バッチサイズを変えてみる（一回の学習のデータ数）
BATCHSIZE_LIST = [2**n for n in range(5, 11)]  # バッチサイズのリスト
EPOCH = 10  # エポック数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用するデバイス（GPU or CPU）

loss_list = []  # 損失の履歴を保存するリスト
accuracy_list = []  # 正解率の履歴を保存するリスト
epoch_list = list(range(EPOCH))  # エポック数のリスト

train_dataset = tv.datasets.MNIST(
    root="./",
    train=True,
    transform=tv.transforms.ToTensor(),
    download=True
)

test_dataset = tv.datasets.MNIST(
    root="./",
    train=False,
    transform=tv.transforms.ToTensor(),
    download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCHSIZE_LIST[-1],
    shuffle=False
)


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.l1 = nn.Linear(784, 300)  # 入力層から中間層への全結合層
        self.l2 = nn.Linear(300, 300)  # 中間層から中間層への全結合層
        self.l3 = nn.Linear(300, 10)  # 中間層から出力層への全結合層

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y


def train(train_loader, optimizer, device):
    start_time = time.perf_counter()  # 学習開始時間を計測
    model = MNIST().to(device)  # モデルをデバイスに転送
    model.train()  # モデルを学習モードに設定
    for epoch in range(EPOCH):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 勾配の初期化
            y = model(images)  # モデルの出力を計算
            loss = F.cross_entropy(y, labels)  # 損失の計算
            total_loss += loss.item()  # トータルの損失を更新
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータの更新

        loss_list.append(float(total_loss))  # 損失をリストに追加
        if epoch == 1:
            print(f"epoch1のときのloss={total_loss}")  # epoch1のときの損失を
