# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:52:16 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# データを何回学習するか
EPOCH = 10
# 学習率
lr=0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trans = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.5,), (0.5,))])

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.l1 = nn.Linear(28*28, 512)  # 入力層784→512 (28x28の画像情報)
        self.l2 = nn.Linear(512, 512)  # 中間層512→512
        self.l3 = nn.Linear(512, 10)  # 最終層512→10
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.l1(x)) #活性化関数のランプ関数に通す
        h = F.relu(self.l2(h)) #負の入力を0,それ以外をそのまま出力する関数
        y = self.l3(h)
        return y


def train(train_dataset, optimizer_name, batch_size, epoch, model_path):
    """
    与えられたデータセットでモデルを学習する関数

    :param train_dataset: 学習用データセット
    :param optimizer_name: 最適化アルゴリズム名 (Adam, SGDのいずれか)
    :param batch_size: バッチサイズ
    :param epoch: 学習回数
    :param model_path: 学習したモデルを保存する場所
    :return: 学習後のモデル
    """
    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # モデルの初期化
    model = MNIST().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Optimizerの設定
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer specified.")

    #学習開始
    for i in range(epoch):
        model.train()
        total_loss = 0

        for images, labels in train_loader:# 重みの調整
            images = images.view(-1, 28 * 28).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # 誤差の逆伝播
            optimizer.step()
            total_loss += loss.item()
        print(f"エポック [{epoch+1}/{i}], Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"{model_path}に学習済みのモデルを保存しました")
    return model


def test(test_dataset, batch_size, model_path): #test
    """
    学習済みモデルを用いて、与えられたテストデータセットで評価する関数

    :param test_dataset: テスト用データセット
    :param batch_size: バッチサイズ
    :param model_path: 読み込むモデルの場所
    :return:
    """
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    state_dict = torch.load(model_path)
    model = MNIST()
    model.load_state_dict(state_dict)
    model.eval()
    total = len(test_loader.dataset)
    correct = 0

    with torch.no_grad():#勾配を計算せずに処理速度を上げる
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"正解率: {accuracy:.4f}")


if __name__ == '__main__':
    batch_size=100
    optimizer_name = 'Adam'
    epoch=8
    model_path=f"model/model_batch{batch_size}_epoch{epoch}_{optimizer_name}.pth"
    train_dataset = tv.datasets.MNIST(root="./", train=True, transform=tv.transforms.ToTensor(), download=True)
    test_dataset = tv.datasets.MNIST(root="./", train=False, transform=tv.transforms.ToTensor(), download=True)
    print(f"BATCHSIZE: {batch_size}")
    print(f"Optimizer: {optimizer_name}")
    print(f"epoch: {epoch}")
    train(train_dataset, optimizer_name, batch_size, epoch, model_path)
    test(test_dataset, batch_size, model_path) 
