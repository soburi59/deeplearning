import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# バッチサイズ (一度の学習におけるデータの数)
BATCHSIZE = 1200
# データを何回学習するか
EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_list = []
accuracy_list = []
epoch_list = list(np.arange(0, EPOCH, 1))

# MNISTデータセットの読み込み
train_dataset = tv.datasets.MNIST(root="./",
                                  train=True,
                                  transform=tv.transforms.ToTensor(),
                                  download=True)

test_dataset = tv.datasets.MNIST(root="./",
                                 train=False,
                                 transform=tv.transforms.ToTensor(),
                                 download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCHSIZE,
                                          shuffle=False)


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.l1 = nn.Linear(784, 300)  # 入力層784→300 (28x28の画像情報)
        self.l2 = nn.Linear(300, 300)  # 中間層300→300
        self.l3 = nn.Linear(300, 10)  # 最終層300→10

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y


def train(train_loader, flag):
    start_time = time.perf_counter()
    model = MNIST().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters()) if flag else torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(EPOCH):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28*28).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            y = model(images)
            loss = F.cross_entropy(y, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        loss_list.append(float(total_loss))
        if epoch == 1:
            print(f"epoch1のときのloss={total_loss}")
    print(f"学習処理時間: {time.perf_counter() - start_time}s")


def test(test_loader, i):
    total = len(test_loader.dataset)
    correct = 0
    model = MNIST().to(DEVICE)
    model.load_state_dict(torch.load(f"epoch_{i}.model"))
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).to(DEVICE)
            labels = labels.to(DEVICE)
            y = model(images)
            pred_labels = y.max(dim=1)[1]
            correct += (pred_labels == labels).sum()
    accuracy_list.append(correct.item() / total)


def save_fig(i, flag):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"BATCHSIZE={i}")
    ax1.set_title("loss_graph")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_ylim(0, 100)
    ax1.plot(loss_list)
    ax1.set_xticks(epoch_list)

    ax2.set_title("accuracy_graph")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.9, 1)
    ax2.plot(accuracy_list)
    ax2.set_xticks(epoch_list)

    fig.tight_layout()
    plt.savefig(f"graph_batch{i}.png")
   


batch_sizes = [2**n for n in range(5, 11)]
print("---------------")
for batch_size in batch_sizes:
    print(f"BATCHSIZE={batch_size}")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    train(train_loader, True)
    for j in range(EPOCH):
        test(test_loader, j)
    save_fig(batch_size, True)
    loss_list = []
    accuracy_list = []

print("---------------")
print("Adam → SGD")
batch_size = 256
print(f"BATCHSIZE={batch_size}")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
train(train_loader, False)
for j in range(EPOCH):
    test(test_loader, j)
save_fig(batch_size, False)
