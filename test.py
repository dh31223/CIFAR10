import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

# 模型类定义（与 main.py 中完全相同，加载时需要）
class model(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 第一块
            nn.Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
            # 第三块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=128),
            # 展平
            nn.Flatten(),
            # 全连接层
            nn.Linear(in_features=2048, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型（结构 + 参数一次性恢复）
    mymodel = torch.load('model/best_model.pth', map_location=device, weights_only=False)
    mymodel.eval()

    # 测试数据（与训练时的测试 transform 完全一致）
    test_dataset = torchvision.datasets.CIFAR10(
        root=r'data', train=False,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=64,
        shuffle=False, drop_last=False
    )

    # 从 10000 个测试样本中随机抽取 5000 个，分成 5 份各 1000
    indices = torch.randperm(len(test_dataset))[:5000]
    subsets = [indices[i*1000 : (i+1)*1000] for i in range(5)]

    acc_list = []
    with torch.no_grad():
        for i, subset_idx in enumerate(subsets):
            subset_loader = DataLoader(
                Subset(test_dataset, subset_idx),
                batch_size=64, shuffle=False
            )
            correct = 0
            for feature, label in subset_loader:
                feature = feature.to(device)
                label = label.to(device)

                predict = mymodel(feature)
                pre_label = predict.argmax(dim=1)
                correct += (pre_label == label).sum().item()

            acc = correct / 1000
            acc_list.append(acc)
            print(f'第{i+1}份 正确率：{acc:.4f} ({correct}/1000)')

    acc_mean = sum(acc_list) / len(acc_list)
    acc_std = (sum((a - acc_mean)**2 for a in acc_list) / len(acc_list)) ** 0.5
    print(f'\n均值：{acc_mean:.4f}  标准差：{acc_std:.4f}')
