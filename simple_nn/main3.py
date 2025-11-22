import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        import numpy as np
        return np.array(self.times).cumsum().tolist()

# 定义输入层大小，隐藏层大小，输出层大小，批量大小
n_in, n_hidden, n_out, batch_size = 10, 35, 1, 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in, dtype=torch.float32, device=device)
y = torch.tensor([
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]
], dtype=torch.float32, device=device)

# 创建顺序模型，包含线性层，ReLU 激活函数 和 sigmoid 激活函数
model = nn.Sequential(
    nn.Linear(n_in, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_out),
    nn.Sigmoid(),
)

model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
# 学习率为 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 用于存储每一轮的损失值
loss_history = []

if __name__ == '__main__':
    # --- 使用方法 ---
    timer = Timer()
    timer.start()
    for epoch in range(100_000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss_history.append(loss.item())
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = timer.stop()
    print(f'Train time: {epoch_time:.4f} sec, on {device}')

# 可视化损失变换曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 100001), loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 可视化预测结果与实际目标值对比
# y_pred_final = model(x).detach().numpy()  # 最终预测值
# y_actual = y.numpy()  # 实际值
#
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
# plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('Actual vs Predicted Values')
# plt.legend()
# plt.grid()
# plt.show()
