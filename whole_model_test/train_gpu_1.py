#将训练模式改为gpu，需要改三个地方：1.网络模型 2.损失函数 3.训练与测试步骤的数据（输入、标注）

import torch.optim
import torchvision.datasets
from torch.utils.data import  DataLoader
import time
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

#定义模型model
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),   #kernel_size=2,步幅未设置时默认与kernelsize一样
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

#下载数据集
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
#获取测试集大小
test_data_size = len(test_data)

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

tudui = Tudui()
#1.设置网络模型为gpu训练模式
tudui = tudui.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
#2.设置损失函数为gpu训练模式
loss_fn = loss_fn.cuda()

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
        #3.1 将数据（输入、标注）设置为gpu模式
        imgs = imgs.cuda()
        targets = targets.cuda()
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
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 3.2 将数据（输入、标注）设置为gpu模式
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()#argmax将输出展平成一维并取得最大索引
            total_accuracy = total_accuracy + accuracy

    print('整体测试集上的loss：{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy/test_data_size))
    total_test_step = total_test_step + 1

    torch.save(tudui, 'tudui_{}.pth'.format(i))
    print('模型已保存')