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
class DiceLoss(nn.Module):
    def __init__(self, weights=None, device='cuda'):
        super(DiceLoss, self).__init__()
        self.weights = None
        if weights is not None:
            self.weights = t.tensor(weights).float().to(
                device)  # convert weights to a float tensor on the specified device

    def forward(self, input, target):
        smooth = 1.
        target = F.one_hot(target, input.size(1)).permute(0, 3, 1, 2).float()
        input = F.softmax(input, dim=1)
        intersection = (input * target).sum(dim=(2, 3))  # sum over (H, W)
        denominator = (input + target).sum(dim=(2, 3))  # sum over (H, W)
        f_score = (2. * intersection + smooth) / (denominator + smooth)  # average over classes and batch
        if self.weights is not None:
            assert len(self.weights) == f_score.size(1), "weights and classes number must be same."
            f_score = f_score * self.weights  # weight each class
        return (1 - f_score.mean())


Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=1, shuffle=True, num_workers=0)

mode = UNet(n_channels=3, n_classes=cfg.NCLSS)
mode = mode.to(device)
# criterion = nn.CrossEntropyLoss().to(device)
criterion = DiceLoss(weights=[0.3, 1.0, 1.0]).to(device)
optimizer = optim.Adam(mode.parameters(), lr=cfg.LR)

writer = SummaryWriter('runs')
try:
    checkpoint = t.load('./logs/epoch_2.pth')
    print('Model loaded, resuming training from epoch {}.'.format(checkpoint['epoch']))
    mode.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
except FileNotFoundError:
    start_epoch = 0
print("开始训练轮次：")
print(start_epoch)
def train(model):
    best = [0]
    net = model.train()
    step = 0
    # 训练轮次
    for epoch in range(start_epoch,cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 10 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.9
        

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        train_dice=0
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))   # [4, 3, 352, 480]
            img_label = Variable(sample['label'].to(device))    # [4, 352, 480]
            # 训练
            out = net(img_data)     # [4, 12, 352, 480]
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()    # (4, 352, 480)
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()   # (4, 352, 480)
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_dice += eval_metrix['DICE']
            train_class_acc += eval_metrix['class_accuracy']
            step=step+1
            writer.add_scalar('loss',loss, step)

            
            if i%50==0:
                print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        
            



        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_loss|:{:}\n\n|Train_dice|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_loss / len(train_data),
            train_dice / len(train_data),
        )


        print(metric_description)
        print("准备保存模型...")
        flag=0
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            # t.save(net.state_dict(),'./logs/{}.pth'.format(epoch+1))
            t.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, './logs/epoch_{}.pth'.format(epoch+1))
            flag=1
            print("保存模型成功！")
        if flag==0:
            print("该模型没必要保存！")

        net = model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0
        eval_dice=0

        prec_time = datetime.now()
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].to(device))
            valLabel = Variable(sample['label'].long().to(device))

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label)
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou
            eval_dice = eval_metrics['DICE'] + eval_dice
        # eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

        val_str = ('|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid loss|:{:}\n|Valid dice|:{:}'.format(
            
            eval_acc / len(val_data),
            eval_miou / len(val_data),
            eval_loss / len(val_data),
            eval_dice / len(val_data)))
        print(val_str)
        print(time_str)
        file_handle2=open('train.csv',mode='a+')
  

        train_acc = ('%f' % (train_acc / len(train_data)))
        train_miou = ('%f' % (train_miou / len(train_data)))
        train_loss = ('%f' % (train_loss / len(train_data)))
        train_dice = ('%f' % (train_dice / len(train_data)))
  
        file_handle2.write('epoch'+','+str(epoch+1)+','+'train_acc'+','+train_acc+','+'train_miou'+','+train_miou+','+'train_loss'+','+train_loss+','+'train_dice'+','+train_dice+'\n'  )
        file_handle2.close()
        file_handle2=open('val.csv',mode='a+')


        eval_acc = ('%f' % (eval_acc / len(val_data)))
        eval_miou = ('%f' % (eval_miou / len(val_data)))
        eval_loss = ('%f' % (eval_loss / len(val_data)))
        eval_dice = ('%f' % (eval_dice / len(val_data)))
  
        file_handle2.write('epoch'+','+str(epoch+1)+','+'val_acc'+','+eval_acc+','+'val_miou'+','+eval_miou+','+'val_loss'+','+eval_loss+','+'val_dice'+','+eval_dice+'\n'  )
        file_handle2.close()

        writer.close()
 



def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou
    # eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(mode)
    evaluate(mode)

