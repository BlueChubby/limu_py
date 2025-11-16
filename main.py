from datetime import datetime
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l

def test_cuda_env()-> None:
    if torch.cuda.is_available():
        print("GPU available")
        print(torch.cuda._version)

    if torch.mps.is_available():
        print("MPS available")


def get_fashion_mnist_labels(labels):
    """ 返回 fashion-mnist 数据集的文本标签 """
    text_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandals", "Shirt",
        "Sneaker", "Bag", "Ankle boot"
    ]
    return [text_labels[int()] for i in labels]

if __name__ == '__main__':

    print(Path.cwd())
    print("torchvision version: ", torchvision.__version__)
    test_cuda_env()

    d2l.use_svg_display()

    trans = transforms.ToTensor()
    print(type(trans))

    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=trans,
        download=True
    )

    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=trans,
        download=True
    )

    print(mnist_train)
    print(len(mnist_train))
    print(len(mnist_test))

    print(mnist_train[0][0].shape)



