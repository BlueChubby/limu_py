# 参数管理, 模型参数保存和加载

import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

X = torch.rand(2, 4)

net(X)

print(net[2].state_dict())

print([(name, param.shape) for name, param in net[2].named_parameters()])

for parameter in net.parameters():
    print(parameter)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, input):
        out = self.fc1(input)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # x = torch.rand(4, 2)
    # y = torch.rand(2, 4)
    # torch.save([x, y], "x-files")
    # x2, y2 = torch.load("x-files")
    # print(x2)
    # print(y2)

    net = MLP()
    print(net.state_dict())
    x = torch.rand(2, 4)
    y = net(x)
    print(y)
    # 保存模型参数
    torch.save(net.state_dict(), 'mlp.params')

    # 加载模型结构
    clone = MLP()
    # 加载模型参数
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()

    # 模型验证
    y_clone = clone(x)
    print(y_clone)

    print(torch.equal(y, y_clone))

