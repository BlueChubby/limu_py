import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        super().__init__()
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y


if __name__ == '__main__':
    print(torch.__version__)

    X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    Y_data = [1, 0, 1, 0]
    # 加载 dataset
    dataset = MyDataset(X_data, Y_data)
    # print(type(dataset))

    # 使用 dataLoader 加载
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    epochs = 3

    for epoch in range(epochs):
        for batch_idx, (features, labels) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}:')
            print(f'Features: {features}')
            print(f'Labels: {labels}')

    # for data_list in dataloader:
    #     print(data_list)
    #     tensor_x = data_list[0]
    #     print(tensor_x)
    #
    #     tensor_y = data_list[1]
    #     print(tensor_y)
