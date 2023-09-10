########################################################### 常用 #######################################################
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device定义

########################################################## 模型保存 #####################################################
network = torch.hub.load()  # network如何定义在网络搭建中有,这里定义好你的network就行

# 方法一：完整保存
torch.save(network, "model/model.pkl")  # 保存网络,保存路径
torch.load("model/model.pkl")  # 读取网络,但是数据量大

# 方法二：保存参数
torch.save(network.state_dict(), "model/params.pkl")  # 数据量小但读取时需要重新获取网络结构
network.load_state_dict()  # 读取数据

########################################################### 数据处理 ####################################################
import re
import numpy as np

data = []
ff = open(".data/.csv/.json/...").readlines()  # 读取数据的所有行单行就是.readline(),当然也可以用os.path.join()等来实现读取
for item in ff:
    out = re.sub(r"\s{2,}", "", item).strip()  # 对所有ff数据进行处理将所有两个以上的空格变为一个空格
    data.append(out.split(","))  # 在列表data里添加数据out,添加时按照数据内的,来隔开

# ---------------- 数据类型转换 -----------------#
data = []
data = np.array(data).astype(np.float)  # 数据类型转换
data = torch.tensor(data)  # 数据类型转换tensor(低级用法，并不推荐会警告)
data = data.clone.detach()  # 先克隆后返回一个全新的tensor (推荐写法)
data = np.array(data).tolist()  # 同tostring,tofile
data[0] = round(3.4861)  # 四舍五入时可以使用round()


########################################################## 网络搭建1(简单) ###############################################
# 网络搭建流程：网络定义->损失函数定义->优化器定义（简单）
# 用于处理文本数据的线性回归模型
class Network(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(Network, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)  # 隐含层使用torch自带包Linear做线性回归
        self.predict = torch.nn.Linear(100, n_out)  # 预测层

    # 简单网络实例在初始化里写好网络层的定义

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.predict(x)
        return x
    # forward为网络的前向传播,x为输入数据。使用self使用定义中的网络实例


# loss
loss_func = torch.nn.MSELoss()  # 使用的是均方误差损失函数

# optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)  # 优化器（网络参数,学习率）因为使用了Adam优化器所以用了较高学习率

# 网络框架调用流程：调用网络的初始化定义->调用网络的前向传播获得预测结果->调用损失函数->损失函数梯度归零->反向传播->单次优化->输出结果

network = Network(18, 1)  # 调用定义网络,18个输入类别,一个输出。使用Linear的线性回归
for i in range(length(data)):  # 针对data内的数据进行循环处理
    pred = network.forward(x_data)  # 调用网络前向传播获得预测结果
    loss = loss_func(pred, y_label)  # 通过损失函数计算预测值和真实label间的损失
    optimizer.zero_grad()  # 损失梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 进行单次优化
    print("epoch:{},loss_train:{}".format(i, loss))  # 显示epoch和损失

########################################################## 网络搭建2(中等) #################################################
# 网络搭建流程：网络定义->损失函数定义->优化器定义（简单）
# 用于手写字母识别的简单识别任务
# Out=(In-Kernel+2Padding)/Stride+1
# torch.summary(vgg19, input_size = [(3, 224, 224)]) 输出网络架构,每一层的输出特征尺寸,及网络参数情况
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

train_data = dataset.MNIST(root="datasets", train=True, download=False, transform=transforms.ToTensor())
# 使用torch.datasets里自带的数据集MNIST,并使用transforms把数据转变为tensor

# batch_size
train_dataloader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)


# 使用utils.data包加载数据,其中使用Shuffle打乱数据
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # 输入size为1*28*28
            # Out=(In-Kernel+2Padding)/Stride+1
            nn.BatchNorm2d(32),  # 正则化
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化
        )

        self.classifier = nn.Sequential(  # 定义一个线性分类器
            nn.Linear(in_features=14 * 14 * 32, out_features=10),
        )

    def forward(self, x):
        x = self.conv(x)  # 卷积操作后的tensor为(n,c,h,w)
        x = x.view(x.size()[0], -1)  # 将n,c,h,w的tensor拉直拉成一个n*n_feature的一个tensor用于线性全连接
        x = self.classifier(x)
        return x
