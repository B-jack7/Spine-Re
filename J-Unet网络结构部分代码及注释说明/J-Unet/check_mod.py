import torch
import torch.nn as nn
from nets.unet3plus import UNet_3Plus

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 创建UNet模型

n_classes = 3
model = UNet_3Plus(n_classes=n_classes)

# 打印模型参数量
print("模型参数量：", count_parameters(model))

