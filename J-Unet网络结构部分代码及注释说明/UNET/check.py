import torch
import torch.nn as nn
from unet.unet_model import UNet
from torchsummary import summary

import torchsummary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 创建UNet模型
n_channels = 3
n_classes = 3
model = UNet(n_channels=n_channels, n_classes=n_classes)

# 打印模型参数量
print("模型参数量：", count_parameters(model))

# 打印模型结构
print(model)
torchsummary.summary(model.cuda(),(3, 512, 512))

