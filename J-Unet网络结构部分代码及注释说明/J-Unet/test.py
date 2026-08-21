import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nets.unet3plus import UNet_3Plus
from evalution_segmentaion import eval_semantic_segmentation
from dataset import CamvidDataset
import cfg
import torch.nn as nn
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# device = t.device('cpu')
BATCH_SIZE = 1
miou_list = [0]

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

net = model =UNet_3Plus(n_classes=cfg.NCLSS)
net.eval()
net.to(device)
net.load_state_dict(t.load('./logs/2.pth'))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0
train_f1score=0
recall=0
dice=0
test_loss=0
criterion = nn.CrossEntropyLoss().to(device)
for i, sample in enumerate(test_data):
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)
	loss = criterion(out, label)
	test_loss = loss.item() + test_loss
	

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]

	true_label = label.data.cpu().numpy()
	true_label = [i for i in true_label]

	eval_metrix = eval_semantic_segmentation(pre_label, true_label)
	train_acc = eval_metrix['mean_class_accuracy'] + train_acc
	train_miou = eval_metrix['miou'] + train_miou
	train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
	train_f1score = eval_metrix['f1score'] + train_f1score
	recall = eval_metrix['recall'] + recall
	dice = eval_metrix['DICE'] + dice
	
	if len(eval_metrix['class_accuracy']) < 2:
		eval_metrix['class_accuracy'] = 0
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']
		error += 1
	else:
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']

	


epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f} ,f1_score:{:.5f}, reall:{:.5f}, dice:{:.5f}, testloss:{:.5f}'.format(train_acc /(len(test_data)-error),
															train_miou/(len(test_data)-error), train_mpa/(len(test_data)-error),train_f1score/(len(test_data)-error),
															recall/(len(test_data)-error),dice/(len(test_data)-error),test_loss/(len(test_data)-error)
															))

# if train_miou/(len(test_data)-error) > max(miou_list):
miou_list.append(train_miou/(len(test_data)-error))
print(epoch_str+'==========last')