import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=64,
    shuffle=True # 训练时一定要打乱
)
# 2. 训练循环 (假设我们训练 5 个周期 Epoch)
num_epochs = 5
for epoch in range(num_epochs):

    print(f"--- 开始第 {epoch + 1} 个周期 ---")

    # 3. 关键区别在这里！
    #    这个 for 循环会 *一直* 运行，直到遍历完 train_loader 里的 *所有* 批次
    #    (60,000 张图 / 64 = 938 个批次)
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):

        # 在这里执行训练：
        # 1. model(batch_features)  # 喂给模型
        # 2. loss = ...             # 计算损失
        # 3. loss.backward()        # 反向传播
        # 4. optimizer.step()       # 更新权重

        if batch_idx % 100 == 0:
            print(f"  已处理 {batch_idx * 64} / 60000 张图片...")

    print(f"--- 第 {epoch + 1} 个周期结束 ---")
