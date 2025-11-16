import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

# if __name__ == '__main__':
#     print(torch.__version__)
    # tensor_a = torch.linspace(start=-1, end=1, steps=50)
    # # print(tensor_a)
    #
    # x = tensor_a.numpy()
    # y1 = 2 * x + 1
    # y2 = x ** 2
    #
    # # 每个 figure 表示一个单独的图片
    # plt.figure()
    # plt.plot(x, y1)
    #
    # plt.figure()
    # plt.plot(x, y2)
    #
    # # 一个 figure 中显示多个函数
    # plt.figure()
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    #
    # # 限制 x 和 y 轴取值范围
    # plt.xlim([0, 1])
    # plt.ylim([0, 3])
    # plt.show()

    # 显示图片
    # tensor_b = torch.randint(0, 255, (3, 3))
    # numpy_b = tensor_b.numpy()
    # print(numpy_b)
    #
    # plt.imshow(numpy_b, cmap='bone', interpolation='nearest', origin="upper")
    # plt.colorbar(shrink=0.8)
    #
    # plt.show()

    # 网格显示
    # fig = plt.figure(figsize=(6, 6))
    #
    # # 2x2 网格中的第 1 个图 (左上)
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.plot([0, 1], [0, 1])  # 画一条线
    # ax1.set_title('图 1 (2, 2, 1)')
    # plt.show()



from pathlib import Path

def gen_test_data(content: str ,file_name: str)-> Path | None:
    data_path = Path(__file__).parent / "test_data"
    if not data_path.exists():
        data_path.mkdir()

    file_path = data_path / file_name
    if file_path.exists():
        print("文件已经存在")
        return file_path

    file_path.write_text(content, encoding="utf-8")
    return file_path


class CustomFileDataset(Dataset):
    """一个自定义的数据集，用于从CSV文件中读取特征和标签。"""

    def __init__(self, file_path):
        # 1. 初始化 (构造函数)
        # -----------------------------
        # 我们在这里只做一次繁重的工作：打开文件并加载所有数据。

        # 我们将使用两个列表来存储内存中的所有数据
        all_features_list = []
        all_labels_list = []

        print(f"开始加载文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用 csv 模块来读取
            reader = csv.reader(f)

            # 跳过第一行（标题行，即 "feature1,feature2,..."）
            next(reader)

            for row in reader:
                # row 是一个列表, e.g., ['1.2', '0.5', '3.1', '0']

                # a) 提取特征 (除最后一列外的所有列)
                #    并转换为 float
                features = np.array(row[:-1], dtype=np.float32)

                # b) 提取标签 (最后一列)
                #    并转换为 int
                label = int(row[-1])

                all_features_list.append(features)
                all_labels_list.append(label)

        # 加载完成后，将 Python 列表转换为 PyTorch Tensors
        # 这是一个重要的优化！
        # 这样 __getitem__ 就可以非常快地直接索引 Tensor
        self.data = torch.tensor(np.array(all_features_list), dtype=torch.float32)

        # 标签通常是 LongTensor (int64)
        self.labels = torch.tensor(np.array(all_labels_list), dtype=torch.long)

        print(f"文件加载完毕。总共 {len(self.labels)} 条数据。")

    def __len__(self):
        # 2. 获取长度
        # -----------------------------
        # 这个函数必须返回数据集的总大小
        return self.labels.shape[0]  # 或者 len(self.labels)

    def __getitem__(self, idx):
        # 3. 获取单条数据
        # -----------------------------
        # 这个函数接收一个索引 idx (从 0 到 len(self)-1)
        # 它必须返回你想要的数据。

        # 因为我们已经在 __init__ 中把所有东西都转成了 Tensor，
        # 这里我们只需要简单地索引它们即可。

        features = self.data[idx]
        label = self.labels[idx]

        # 以元组 (tuple) 形式返回，就像 FashionMNIST 一样
        # (数据, 标签)
        return features, label


if __name__ == '__main__':
    csv_content = """feature1,feature2,feature3,label
    1.2,0.5,3.1,0
    0.9,1.1,2.5,1
    3.4,3.1,1.0,0
    5.0,2.2,0.5,2
    4.1,1.9,0.8,2
    2.2,2.1,2.8,1
    """
    data_file_path = gen_test_data(csv_content, file_name="test.csv")

    # 1. 实例化我们的数据集
    #    __init__ 函数会在这里被调用
    my_dataset = CustomFileDataset(file_path=data_file_path)

    print("-" * 20)

    # 2. 检查长度
    #    __len__ 函数会在这里被调用
    print(f"数据集的总长度: {len(my_dataset)}")

    print("-" * 20)

    # 3. 检查第一条数据
    #    __getitem__(0) 会在这里被调用
    features, label = my_dataset[0]
    print(f"第一条数据 (索引 0):")
    print(f"  特征: {features}")
    print(f"  标签: {label}")
    print(f"  特征的 Shape: {features.shape}")

    print("-" * 20)

    # 4. (最重要) 将它放入 DataLoader
    #    DataLoader 会在后台自动调用 __getitem__ 来批量组合数据
    #    我们将 batch_size 设为 2，所以它每次会取出 2 条数据
    train_loader = DataLoader(
        dataset=my_dataset,
        batch_size=2,
        shuffle=True  # 打乱数据
    )

    # 5. 遍历 DataLoader
    print("开始从 DataLoader 中遍历一个批次 (batch)...")
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        print("\n--- 这是一个新的批次 ---")
        print(f"批次索引 (batch_idx): {batch_idx}")

        print(f"批次特征 (batch_features) 的 Shape: {batch_features.shape}")
        print("批次特征内容:")
        print(batch_features)

        print(f"批次标签 (batch_labels) 的 Shape: {batch_labels.shape}")
        print("批次标签内容:")
        print(batch_labels)

        # 我们只看第一个批次作为演示
        break



