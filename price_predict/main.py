import re

import numpy as np
import torch

# data
data = []
ff = open("kc_house_data.csv").readlines()
for item in ff:
    out = re.sub(r"\s{2,}", ",", item).strip()
    data.append(out.split(","))
data = np.array(data).astype(np.float)
# data由array转float

Y = data[:, -1]  # 最后一栏为price即Y
X = data[:, 0:-1]  # 输入为第一项直到最后除了最后一栏
length = Y.shape  # length[0]即有多少行数据,按照9:1划分训练集测试集
train_length = round(length[0] * 0.9)

Y_train = Y[0:train_length, ]  # 训练集
X_train = X[0:train_length, ]

Y_test = Y[train_length:length[0], ]  # 测试集
X_test = X[train_length:length[0], ]


# network
class Network(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(Network, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_out)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.predict(x)
        return x


network = Network(18, 1)  # 调用定义网络，18个输入类别，一个输出。使用Linear的线性回归

# loss
loss_func = torch.nn.MSELoss()  # 使用的是均方误差损失函数

# optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)  # 优化器（网络参数，学习率）因为使用了Adam优化器所以用了较高学习率

# training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)

    pred = network.forward(x_data)  # 前向传播.这里需要注意network=Network(18, 1)和network.forward(x_data)都是需要的
    pred = torch.squeeze(pred)  # 删除一个维度
    loss = loss_func(pred, y_data) * 0.001  # 计算损失
    optimizer.zero_grad()  # 损失梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 进行单次优化

    # print("epoch:{},loss_train:{}".format(i, loss))  # 显示epoch和损失
    # print(pred[0:10])
    # print(y_data[0:10])

    # testing
    x_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(Y_test, dtype=torch.float32)

    pred_test = network.forward(x_test)  # 前向传播.这里需要注意network=Network(18, 1)和network.forward(x_data)都是需要的
    pred_test = torch.squeeze(pred_test)  # 删除一个维度
    loss_test = loss_func(pred_test, y_test) * 0.001  # 计算损失
    optimizer.zero_grad()  # 损失梯度归零
    loss_test.backward()  # 反向传播
    optimizer.step()  # 进行单次优化
    print("epoch:{},loss_train:{},loss_test:{},pred_train:{},y_train:{},pred_test:{},y_test:{}".format(i, loss, loss_test, pred[0], y_data[0],
                                                                                                     pred_test[0], y_test[0]))  # 显示epoch和损失

# 保存训练模型
# torch.save(network, "model/model.pkl")
