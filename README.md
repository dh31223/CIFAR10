

# CIFAR-10 图像分类

从头手写 CNN 在 CIFAR-10 上达到 **84.6%** 测试准确率，不使用预训练模型。

## 项目概览

| 项目 | 说明 |
|------|------|
| 数据集 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 10 类，32×32 彩色图，5 万训练 + 1 万测试 |
| 任务 | 图像分类（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车） |
| 框架 | PyTorch |
| 模型 | 3 卷积块 + 全连接分类头（自建，无预训练） |
| 最佳准确率 | **84.6%**（5 折交叉验证均值 84.44% ± 0.83%） |
| 训练设备 | NVIDIA GPU（CUDA） |

## 模型结构

```
输入 (3×32×32)
  ↓
Conv2d(3→32, 3×3) → BN → ReLU → MaxPool(2×2)    # 32×32 → 16×16
  ↓
Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)   # 16×16 → 8×8
  ↓
Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2×2)  # 8×8 → 4×4
  ↓
Flatten → 2048
  ↓
Linear(2048→256) → BN1d → ReLU
  ↓
Linear(256→10)
```

## 关键优化路径

| 阶段 | 测试准确率 | 改动 |
|------|:---:|------|
| 裸奔 CNN | 37% | 2 卷积块 + 全连接，无 BN，无增强 |
| + BatchNorm | 65% | 每层卷积后加 BN |
| + Normalize | 68% | 像素归一化到 CIFAR-10 标准均值/方差 |
| + 数据增强 | 74% | RandomFlip + RandomCrop + RandomRotation |
| + 第三卷积块 | 76% | 通道数 32→64→128 |
| + AdamW | 77.5% | SGD → AdamW |
| + 30 epochs | **84.6%** | epochs 15 → 30 |

## 数据增强策略

仅使用测试集里**会出现的变化**：

- `RandomHorizontalFlip(p=0.3)` — 水平翻转
- `RandomCrop(size=32, padding=4)` — 先填充到 40×40 再随机裁回 32×32
- `RandomRotation(degrees=15)` — 小角度旋转

不使用 `RandomVerticalFlip`（CIFAR 不会倒着拍）。

## 文件说明

```
new_project/
├── main.py              # 训练脚本（含模型定义、数据加载、训练循环、TensorBoard 日志）
├── test.py              # 测试脚本（5 折 × 1000 样本，输出均值和标准差）
├── test1.py             # 单张图片推理脚本（输入图片→输出类别名称）
├──  best_model.pth   # 最佳模型权重（torch.save 完整模型）
└── README.md
```

## 快速复现

### 环境

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tensorboard

### 训练

```bash
cd new_project
python main.py
```

训练日志写入 `logs/`，启动 TensorBoard 查看：

```bash
tensorboard --logdir=logs
```

### 测试

```bash
python test.py
```

输出示例：

```
第1份 正确率：0.8480 (848/1000)
第2份 正确率：0.8360 (836/1000)
第3份 正确率：0.8520 (852/1000)
第4份 正确率：0.8330 (833/1000)
第5份 正确率：0.8530 (853/1000)

均值：0.8444  标准差：0.0083
```

### 单张推理

```bash
python test1.py
```

将待测图片放至 `data/` 目录并修改 `test1.py` 中的图片路径即可。

## 踩坑记录

完整的问题排查与解决方案见个人学习笔记（DL.md §10.1），涵盖：
- 数据加载（Dataset/Transform/DataLoader 关系）
- 数据增强原则（只增强测试集会出现的变化）
- BN 放置位置（为什么 Conv→BN→ReLU 而非 Conv→ReLU→BN）
- BatchNorm2d vs BatchNorm1d 的选择
- GPU 训练优化（num_workers / pin_memory / non_blocking）
- 过拟合 vs 欠拟合的判断方法

## 作者

- 大二本科生，AI Infra 方向学习路线中
- 本项目为 PyTorch 独立实战第一阶段（CIFAR-10）




