import torch
import torch.optim as optim
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Linear(2, 4)
        # 定义激活函数
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 定义模型
    model = MyNet()
    # 定义loss
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters())

    # 数据集
    X = torch.rand(10, 2)
    Y = torch.rand(10, 1)

    for epoch in range(100):
        # 定义模型模式
        model.train()
        # 梯度归零
        optimizer.zero_grad()
        # 计算损失值
        loss = criterion(model(X), Y)
        # 计算梯度 （loss 对应每一个模型权重参数的偏导数）
        loss.backward()
        # 优化器根据梯度更新模型参数
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # 每 10 轮输出一次损失
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')




