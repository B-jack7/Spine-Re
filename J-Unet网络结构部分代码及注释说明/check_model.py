import torch
from torchsummary import summary
from nets.unet3plus import UNet_3Plus
import cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型并加载到设备
model = UNet_3Plus(n_classes=cfg.NCLSS)
model = model.to(device)

# 输入维度，例如(3, 352, 480)，可以根据实际情况修改
input_size = (3, 352, 480)

# 打印模型结构和参数量
summary(model, input_size=input_size)
