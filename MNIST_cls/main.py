import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset
# train_data = dataset.CocoDetection(root="D:/Project/datasets/coco/images/val",
#                                    annFile="D:/Project/datasets/coco/annotations/instances_val2017.json",
#                                    transform=transforms.ToTensor())
#
# test_data = dataset.CocoDetection(root="D:/Project/datasets/coco/images/val",
#                                   annFile="D:/Project/datasets/coco/annotations/instances_val2017.json",
#                                   transform=transforms.ToTensor())

train_data = dataset.MNIST(root="datasets", train=True, download=False, transform=transforms.ToTensor())

test_data = dataset.MNIST(root="datasets", train=False, download=False, transform=transforms.ToTensor())

# batch_size
train_dataloader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# 使用utils.data包加载数据,其中使用Shuffle打乱数据
test_dataloader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


# network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # 输入size为1*28*28
            # (picture+ 2*padding -k_size)/stride+1
            nn.BatchNorm2d(32),  # 正则化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化
        )

        self.classifier = nn.Sequential(  # 定义一个线性分类器
            nn.Linear(in_features=14 * 14 * 32, out_features=10),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(in_features=6272, out_features=10),
        )

    def forward(self, x):
        x = self.conv(x)  # 卷积操作后的tensor为(n,c,h,w)
        x = x.view(x.size()[0], -1)  # 将n,c,h,w的tensor拉直拉成一个n*n_feature的一个tensor用于线性全连接
        x = self.classifier(x)
        return x


network = Network()
network = network.cuda()
# loss
loss_func = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(params=network.parameters(), lr=0.01)

# training
for epoch in range(20):
    for i, (images, labels) in enumerate(train_dataloader):  # 在每个epoch中使用enumerate(*)---实现遍历/遍历*的内容
        # 注意这里遍历的是dataloader因为data_utils给原先的数据(n,h,w)增加了batch_size变成(b,n,h,w)
        images = images.cuda()
        labels = labels.cuda()

        prediction = network(images).to(device)  # 让图片跑网络得到预测结果
        loss = loss_func(prediction, labels)  # 对预测结果求取损失函数

        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度

        # print("epoch:{}, cur_item:{}/{}, loss:{}".format(epoch + 1, i, len(train_data) // 4, loss.item()))
        # 输出epoch,当前的item并使用loss.item获取当前loss值

    # eval
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_dataloader):  # 在每个epoch中使用enumerate(*)---实现遍历/遍历*的内容
        images = images.cuda()  # 使用cuda跑
        labels = labels.cuda()
        predictions = network(images)

        # prediction = batch_size * num_class
        loss_test += loss_func(predictions, labels)  # 计算总共的loss
        _, result = predictions.max(1)
        accuracy += (result == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss = loss_test / (len(test_data) // 64)  # 这里后面要//64是因为在一开始data_loader的时候变成了b,n,h,w即loss_test已经//64
    print("epoch:{},loss:{}, accuracy:{}".format(epoch + 1, loss.item(), accuracy))

# save
torch.save(network, "model/best.pkl")
