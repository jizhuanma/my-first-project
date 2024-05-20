import torch.optim
import torchvision.datasets
from torch.utils.data import  DataLoader
from model import *
import time

#下载数据集
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

tudui = Tudui()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录每轮的训练次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10

start_time = time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始----------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)       #输出为什么是“得分”？
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #梯度优化
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print('训练次数：{}，loss:{}'.format(total_train_step,loss))

    #测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print('整体测试集上的loss：{}'.format(total_test_loss))