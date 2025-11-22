import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        super(MyDataset, self).__init__()

        self.X_data = x_data
        self.Y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx) -> tuple:
        x_tensor = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x_tensor, y_tensor


X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [2, 4, 6, 8]  # 目标标签

# 创建自定义个的数据集
my_dataset = MyDataset(X_data, Y_data)

# 创建 dataLoader 实力
dataLoader = DataLoader(my_dataset, batch_size=2, shuffle=True)

if __name__ == '__main__':

    for idx, (features, labels) in enumerate(dataLoader):
        print(idx)
        print(features.shape)
        print(features)
        print(labels.shape)
        print(labels)
