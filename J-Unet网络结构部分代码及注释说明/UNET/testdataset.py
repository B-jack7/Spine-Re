import torch as t
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation
from tensorboardX import SummaryWriter
from unet.unet_model import UNet
import cfg
#######GPU#########
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

##########CPU#########
# device = t.device('cpu')

Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=1, shuffle=True, num_workers=0)


mode = UNet(n_channels=3, n_classes=cfg.NCLSS)
mode = mode.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(mode.parameters(), lr=cfg.LR)

writer = SummaryWriter('runs')