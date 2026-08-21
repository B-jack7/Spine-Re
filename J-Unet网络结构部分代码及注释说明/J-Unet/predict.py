from typing import List
import pandas as pd
import numpy as np
import torch as t
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from dataset1 import CamvidDataset
import cfg
import os
from nets.unet3plus import UNet_3Plus
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# device = t.device('cpu')

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=1, shuffle=True, num_workers=0)
# net =BiSeNetV1(n_classes=12).to(device)
net =  UNet_3Plus(n_classes=cfg.NCLSS).to(device)
net.load_state_dict(t.load("logs/2.pth"))
net.eval()

pd_label_color = pd.read_csv('./data/class_dict1.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = "./result_pics/"
transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 0, 255],  # cornsilk
        2: [2, 255, 0],  # cornflowerblue
        3: [255, 255, 0],  # mediumAquamarine

    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


imgs = os.listdir("./data/test")
for i in imgs:
	img = Image.open("./data/test/"+i)

	# img=img.resize(cfg.crop_size)

	img=img.convert('RGB')
	img = transform(img).unsqueeze(0) # To tensor

	img = img.to(device)
	out = net(img)
	# print(out.shape)
	out = F.log_softmax(out, dim=1)
	# print(out.shape)
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()







	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	# pre1=pre1.reszie((1440,1440))
	pre1.save(dir + i)
	print('Done')