import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 【核心配置】中文字体设置 (Mac/Windows 通用)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ==========================================

# -----------------------------------------
# 准备数据
# -----------------------------------------
# 1. 回归任务数据：模拟 预测值 与 真实值(0) 的偏差
# 范围从 -3 到 3
x_diff = torch.linspace(-3, 3, 200)
target_reg = torch.zeros_like(x_diff) # 假设真实值是 0

# 2. 分类任务数据：模拟 预测概率 (0~1之间)
# 也就是模型输出 sigmoid 之后的值
p_prob = torch.linspace(0.001, 0.999, 200)
target_cls_1 = torch.ones_like(p_prob)  # 假设真实标签是 1
target_cls_0 = torch.zeros_like(p_prob) # 假设真实标签是 0

# -----------------------------------------
# 计算损失
# -----------------------------------------
# 1. MSE (L2) Loss: (y - y_hat)^2
loss_mse = F.mse_loss(x_diff, target_reg, reduction='none')

# 2. L1 (MAE) Loss: |y - y_hat|
loss_l1 = F.l1_loss(x_diff, target_reg, reduction='none')

# 3. Huber Loss: 结合了 MSE 和 L1
# delta=1.0: 误差小于1用平方(MSE)，大于1用线性(L1)
loss_huber = F.huber_loss(x_diff, target_reg, reduction='none', delta=1.0)

# 4. BCE Loss: -[y*log(p) + (1-y)*log(1-p)]
# 计算当真实标签为 1 时的 Loss
loss_bce_1 = F.binary_cross_entropy(p_prob, target_cls_1, reduction='none')
# 计算当真实标签为 0 时的 Loss
loss_bce_0 = F.binary_cross_entropy(p_prob, target_cls_0, reduction='none')


# -----------------------------------------
# 开始绘图
# -----------------------------------------
plt.figure(figsize=(14, 10))

# --- 图 1: MSE Loss ---
plt.subplot(2, 2, 1)
plt.plot(x_diff.numpy(), loss_mse.numpy(), color='red', linewidth=2, label='MSE (L2)')
plt.title('MSE Loss (均方误差)', fontsize=12, fontweight='bold')
plt.xlabel('预测误差 (Pred - True)')
plt.ylabel('Loss 值')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0, 7, "特点: 误差越大，惩罚越狠\n缺点: 容易受异常值影响", ha='center')

# --- 图 2: L1 Loss ---
plt.subplot(2, 2, 2)
plt.plot(x_diff.numpy(), loss_l1.numpy(), color='blue', linewidth=2, label='L1 (MAE)')
plt.title('L1 Loss (平均绝对误差)', fontsize=12, fontweight='bold')
plt.xlabel('预测误差 (Pred - True)')
plt.ylabel('Loss 值')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0, 2.5, "特点: 梯度恒定\n优点: 对异常值鲁棒", ha='center')

# --- 图 3: Huber Loss ---
plt.subplot(2, 2, 3)
plt.plot(x_diff.numpy(), loss_huber.numpy(), color='purple', linewidth=2, label='Huber (delta=1)')
# 为了对比，把 MSE 和 L1 也画上去淡淡的背景
plt.plot(x_diff.numpy(), 0.5 * x_diff.numpy()**2, 'r--', alpha=0.3, label='MSE参考')
plt.plot(x_diff.numpy(), x_diff.abs().numpy() - 0.5, 'b--', alpha=0.3, label='L1参考')
plt.title('Huber Loss (平滑 L1)', fontsize=12, fontweight='bold')
plt.xlabel('预测误差 (Pred - True)')
plt.ylabel('Loss 值')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0, 3.5, "特点: 中间是抛物线(可导)\n两头是直线(抗噪)", ha='center')

# --- 图 4: BCE Loss ---
plt.subplot(2, 2, 4)
plt.plot(p_prob.numpy(), loss_bce_1.numpy(), color='green', linewidth=2, label='真实标签 y=1')
plt.plot(p_prob.numpy(), loss_bce_0.numpy(), color='orange', linestyle='--', linewidth=2, label='真实标签 y=0')
plt.title('BCE Loss (二元交叉熵)', fontsize=12, fontweight='bold')
plt.xlabel('模型预测概率 (Probability)')
plt.ylabel('Loss 值')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.5, 3, "特点: 越确信越错，惩罚无穷大\n(-log x)", ha='center')

plt.tight_layout()
plt.show()