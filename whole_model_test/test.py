import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

image_path = "../dog.PNG" # ../表明文件在上层目录中
image = Image.open(image_path)

# dog.png是4通道图片，因此需要先转换为RGB的3通道图片
image = image.convert("RGB")
#图像大小调整为 可用于model的(32, 32)，并转换为PyTorch的张量（tensor）形式
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                              torchvision.transforms.ToTensor()])
image = transform(image)

#因为保存模型的方法为方式1，故需要导入model才可以加载模型
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

#加载模型，并且设定在CPU上加载模型
model = torch.load("tudui_29.pth", map_location=torch.device('cpu'))
#模型model的输入要求为四维张量，其形状为 (batch_size, channels, height, width)。
image = torch.reshape(image,(1, 3, 32, 32))
#将模型设置为评估模式
model.eval()

with torch.no_grad():
    output = model(image)
print (output)
print(output.argmax(1))
