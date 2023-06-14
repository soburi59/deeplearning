# -*- coding: utf-8 -*-
#元は愛媛大二宮先生のコードです。転載しない。
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#バッチを変えてみる(一回の学習のデータ数)
BATCHSIZE = 1200
#1つのデータを何回学習するか
EPOCH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
losslist=[]
accuracylist=[]
epochlist=list(np.arange(0,EPOCH,1))
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

# for i in range(10):
#     print(train_dataset[i])
#     plt.imshow(train_dataset[i][0][0], cmap="gray")
#     txt = "label:"+str(train_dataset[i][1])
#     plt.text(2,2, txt, color="white")
#     plt.show()

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST,self).__init__()
        #入力層784→300(28*28 画像の情報)
        self.l1 = nn.Linear(784, 300)
        #中間層300→300
        self.l2 = nn.Linear(300, 300)
        #最終層300→10
        self.l3 = nn.Linear(300, 10)
    def forward(self,x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y

def train(train_loader,flag):
    start_time=time.perf_counter()
    model = MNIST().to(DEVICE)
    #SDGなどに変えてみる
    #重みの修正
    if flag:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
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
        #print("epoch", epoch, ": loss", total_loss)
        torch.save(model.state_dict(), f"epoch_{epoch}.model")
        losslist.append(float(total_loss))
        if epoch == 1:
            print(f"epoch1のときのloss={total_loss}")
    print(f"学習処理時間:{time.perf_counter()-start_time}s")


def test(test_loader,i):
    total = len(test_loader.dataset)
    correct = 0
    model = MNIST().to(DEVICE)
    model.load_state_dict(torch.load(f"epoch_{i}.model"))
    model.eval()
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(DEVICE)
        labels = labels.to(DEVICE)
        y = model(images)
        pred_labels = y.max(dim=1)[1]
        correct += (pred_labels == labels).sum()
    # print("correct:", correct.item())#正解
    # print("total:", total)#全部
    # print("accuracy:", correct.item()/total)#正答率
    accuracylist.append(correct.item()/total)

def save_fig(i,flag):
    fig = plt.figure()
    ax1=fig.add_subplot(121,title="loss_graph")
    ax2=fig.add_subplot(122,title="accuracy_graph")
    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    ax1.set_ylim(0,100)
    ax2.set_ylim(0.9,1)
    ax1.plot(losslist)
    ax2.plot(accuracylist)
    ax1.set_xticks(epochlist)
    ax2.set_xticks(epochlist)
    plt.suptitle(f'BATCHSIZE={i}')
    fig.tight_layout()
    if flag:
        plt.savefig(f"graph_batch{i}.png")
    else:
        plt.savefig(f"graph_batch{i}_SGD.png")

for i in [2**n for n in range(5,11)]:
    print("---------------")
    print(f"BATCHSIZE={i}")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=i,
                                                shuffle=True)
    train(train_loader,True)
    for j in range(EPOCH):
        test(test_loader,j)
    save_fig(i,True)
    losslist=[]
    accuracylist=[]
print("---------------")
print("Adam→SGD")
print(f"BATCHSIZE=256")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True)
train(train_loader,False)
for j in range(EPOCH):
    test(test_loader,j)
save_fig(256,False)