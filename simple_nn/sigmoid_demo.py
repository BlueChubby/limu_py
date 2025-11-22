import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 【核心修复】配置中文字体
# ==========================================
# 1. 设置字体：Mac 优先尝试 'Arial Unicode MS'，Windows 优先 'SimHei'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'Microsoft YaHei', 'sans-serif']

# 2. 解决负号显示问题：
# 修改字体后，坐标轴上的负号 '-' 有时会变方块，设为 False 可修复
plt.rcParams['axes.unicode_minus'] = False
# ==========================================


# 1. 准备数据：生成从 -6 到 6 的数据
x = torch.linspace(-6, 6, 200)

# 2. 计算 6 种激活函数
y_sigmoid = torch.sigmoid(x)
y_tanh = torch.tanh(x)
y_relu = torch.relu(x)
y_leaky = F.leaky_relu(x, negative_slope=0.1)
y_gelu = F.gelu(x)
y_silu = F.silu(x)

# 3. 设置画布 (2行3列)
plt.figure(figsize=(15, 8))


# 通用绘图函数
def plot_activation(idx, x, y, name, color, range_lines=None):
    plt.subplot(2, 3, idx)
    plt.plot(x.numpy(), y.numpy(), label=name, color=color, linewidth=2)

    # 画坐标轴
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)  # x轴
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)  # y轴

    # 画辅助线
    if range_lines:
        for line_y in range_lines:
            plt.axhline(y=line_y, color='gray', linestyle=':', alpha=0.5)

    plt.title(name, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')


# --- 开始绘图 ---

# 1. Sigmoid
plot_activation(1, x, y_sigmoid, 'Sigmoid', 'green', [0, 1])
plt.text(-5, 0.8, "输出: (0, 1)\n二分类首选", fontsize=9)

# 2. Tanh
plot_activation(2, x, y_tanh, 'Tanh', 'purple', [-1, 1])
plt.text(-5, 0.8, "输出: (-1, 1)\n零中心化", fontsize=9)

# 3. ReLU
plot_activation(3, x, y_relu, 'ReLU', 'blue')
plt.text(-5, 4, "主流隐藏层默认\n负区死寂", fontsize=9)

# 4. Leaky ReLU
plot_activation(4, x, y_leaky, 'Leaky ReLU', 'orange')
plt.text(-5, 4, "负区微弱导数\n防止神经元死亡", fontsize=9)

# 5. GELU
plot_activation(5, x, y_gelu, 'GELU', 'red')
plt.text(-5, 4, "Transformer标配\n平滑非线性", fontsize=9)

# 6. SiLU / Swish
plot_activation(6, x, y_silu, 'SiLU (Swish)', 'teal')
plt.text(-5, 4, "Llama/YOLO首选\n自门控特性", fontsize=9)

plt.tight_layout()
plt.show()