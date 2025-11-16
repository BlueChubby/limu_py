import torch

if __name__ == '__main__':
    print(torch.__version__)
    tensor_a = torch.randint(0, 10, (3, 4))
    print(tensor_a)

    # tensor to numpy
    numpy_a = tensor_a.numpy()
    print(numpy_a)
    print(type(numpy_a))

    # numpy to tensor
    tensor_b = torch.from_numpy(numpy_a)
    print(tensor_b)
    print(type(tensor_b))

    tensor_c = torch.linspace(start=-1, end=1, steps=50)
    print(tensor_c)
    print(type(tensor_c))







