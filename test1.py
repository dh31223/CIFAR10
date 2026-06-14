
import torch
from torchvision.transforms import v2
import torch.nn as nn
from PIL import Image

ans_list = {0: '飞机', 1: '汽车', 2: '鸟', 3: '猫', 4: '鹿', 5: '狗', 6: '青蛙', 7: '马', 8: '船', 9: '卡车'}

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


#导入模型
mymodel = torch.load(f = r'model\best_model.pth', weights_only = False)

img = Image.open(fp = r'data\老鼠.png')


transform = v2.Compose([
    v2.Resize((32, 32)), 
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

img = transform(img)
img = img.reshape((1, 3, 32, 32))
img = img.cuda()

result = ans_list[mymodel(img).argmax(dim = 1).item()]

print(result)

