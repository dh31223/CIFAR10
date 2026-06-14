import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torchvision.transforms import v2


# 模型声明
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
    # 参数区
    epochs = 30
    batch_size = 64
    learn_rate = 0.001

    # 数据导入
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomCrop(size=32, padding=4),
        v2.RandomRotation(degrees=15),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=r'data', train=True, transform=train_transforms)

    test_dataset = torchvision.datasets.CIFAR10(
        root=r'data', train=False,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    )

    train_len = len(train_dataset)
    test_len = len(test_dataset)

    # 打包为 DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=False, num_workers=2, pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True
    )

    write = SummaryWriter('logs')

    # 模型、损失函数、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mymodel = model().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=learn_rate, weight_decay = 1e-4)

    # 开始训练
    os.makedirs('model', exist_ok=True)
    train_step = 1
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        # 训练
        mymodel.train()
        time_state = time.time()
        print('=' * 10, f'第{epoch + 1}次训练开始', '=' * 10)
        for feature, label in train_dataloader:
            feature = feature.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            predict = mymodel(feature)
            l = loss(predict, label)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            write.add_scalar(tag='train_loss', scalar_value=l.item(), global_step=train_step)
            train_step += 1

        time_end = time.time()
        print(f'第{epoch + 1}轮训练耗时 {time_end - time_state:.1f}s')

        # 测试
        mymodel.eval()
        correct = 0
        test_loss_sum = 0
        with torch.no_grad():
            for feature, label in test_dataloader:
                feature = feature.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                predict = mymodel(feature)

                test_loss_sum += loss(predict, label).item()
                pre_label = predict.argmax(dim=1)
                correct += (pre_label == label).sum().item()

        test_loss_avg = test_loss_sum / len(test_dataloader)
        test_acc = correct / test_len
        print(f'第{epoch + 1}轮测试 loss={test_loss_avg:.4f}  正确率={test_acc:.3f}')
        write.add_scalar(tag='test_loss', scalar_value=test_loss_avg, global_step=epoch + 1)
        write.add_scalar(tag='test_acc', scalar_value=test_acc, global_step=epoch + 1)

        # 保存最佳模型（结构 + 参数）
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(mymodel, 'model/best_model.pth')
            print(f'  ↑ 新纪录！已保存 model/best_model.pth')

    write.close()
    print(f'训练完成！最佳模型：第{best_epoch}轮 正确率={best_acc:.3f} → model/best_model.pth')
    print('end!!!!!!!!!!!!!!!!!!!!!!')
