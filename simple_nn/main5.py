from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

# 定义数据预处理的流水线
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # 将图像调整为 128x128
#     transforms.ToTensor(),  # 将图像转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
# ])

# 图像数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(90),  # 随机旋转 90 度
    transforms.RandomResizedCrop(128),  # 随机裁剪并调整为 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':

    project_path = Path(__file__).parent.parent
    image_path = project_path / "images" / "img.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    print(image_path.name)
    print(image_path.suffix)
    image = Image.open(image_path).convert('RGB')

    # 应用预处理
    image_tensor = transform(image)
    # 输出张量的形状
    print(image_tensor.shape)

    plt.figure()
    plt.title("new image")
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.show()