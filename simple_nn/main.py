import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    #  创建一个小模型
    model = SimpleNN()
    print(model)

    input_tensor = torch.randn(1, 2)

    # Relu 激活函数
    output = torch.relu(input_tensor)

    # Sigmoid 激活函数
    output = torch.sigmoid(input_tensor)

    # Tanh 激活函数
    output = torch.tanh(input_tensor)

    # 均方差损失函数
    criterion = nn.MSELoss()

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 二分类交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()

    import torch.optim as optim

    # 使用 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train(mode=True)

