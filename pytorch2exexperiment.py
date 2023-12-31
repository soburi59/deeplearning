# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:52:16 2023
cc
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# データを何回学習するか
EPOCH = 10
# 学習率
lr=0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_list = list(np.arange(0, EPOCH, 1))


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

def train_and_test(optimizer_name, batch_size, epochs):
    loss_list = []
    accuracy_list = []

    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
    
    #train
    for epoch in range(epochs):
        start_time = time.perf_counter()
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

        loss_list.append(total_loss)
        
        #test
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
        accuracy_list.append(accuracy)
        #トレーニングしたモデルの出力
        torch.save(model.state_dict(), f"model/model_batch{batch_size}_epoch{epoch+1}_{optimizer_name}.pth")
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.4f}, Loss: {total_loss:.4f}, Time: {time.perf_counter() - start_time:.2f}s")
    save_fig(batch_size, optimizer_name, loss_list, accuracy_list)


def save_fig(batch_size, optimizer_name, loss_list, accuracy_list):
    # グラフの作成
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"BATCHSIZE={batch_size}, Optimizer={optimizer_name}")

    # Lossグラフの設定
    ax1.set_title("Loss Graph")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 100)
    ax1.plot(loss_list)
    ax1.set_xticks(epoch_list)

    # Accuracyグラフの設定
    ax2.set_title("Accuracy Graph")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.9, 1)
    ax2.plot(accuracy_list)
    ax2.set_xticks(epoch_list)

    # グラフのレイアウト調整
    fig.tight_layout()
    # グラフの保存
    plt.savefig(f"graph/graph_batch{batch_size}_{optimizer_name}.png")


if __name__ == '__main__':
    # MNISTデータセットの読み込み
    trans = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.5,), (0.5,))])
    train_dataset = tv.datasets.MNIST(root="./", train=True, transform=tv.transforms.ToTensor(), download=True)
    test_dataset = tv.datasets.MNIST(root="./", train=False, transform=tv.transforms.ToTensor(), download=True)

    batch_sizes = [2 ** n for n in range(5, 11)]
    optimizers = ['Adam', 'SGD']
    print("---------------")
    for batch_size in batch_sizes:
        print(f"BATCHSIZE={batch_size}")
        for optimizer in optimizers:
            print(f"Optimizer: {optimizer}")
            train_and_test(optimizer, batch_size, EPOCH)
        print("---------------")