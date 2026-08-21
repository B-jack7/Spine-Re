# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers import unetConv2
from nets.init_weights import init_weights

'''
    UNet 3+
'''


class UNet_3Plus(nn.Module):

    def __init__(self, n_classes):
        super(UNet_3Plus, self).__init__()
        # self.args = args
        in_channels = 3
        n_classes = n_classes
        feature_scale = 4
        is_deconv = True
        is_batchnorm = True
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)  # 64->128
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)  # 128->256
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = 360 # 5*64=320

        '''stage 7d'''
        # h3->128*128, hd4->64*64, Pooling 4 times
        self.h3_PT_hd7 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd7_conv = nn.Conv2d(filters[2], 120, 3, padding=1)
        self.h3_PT_hd7_bn = nn.BatchNorm2d(120)
        self.h3_PT_hd7_relu = nn.ReLU(inplace=True)


        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd7_conv = nn.Conv2d(filters[3],120, 3, padding=1)
        self.h4_Cat_hd7_bn = nn.BatchNorm2d(120)
        self.h4_Cat_hd7_relu = nn.ReLU(inplace=True)


        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.h5_UT_hd7 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.h5_UT_hd7_conv = nn.Conv2d(filters[4], 120, 3, padding=1)
        self.h5_UT_hd7_bn = nn.BatchNorm2d(120)
        self.h5_UT_hd7_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv7d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn7d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu7d_1 = nn.ReLU(inplace=True)

        '''stage 6d'''
        # h3->128*128, hd6->96*96, Pooling 4 times
        self.h3_PT_hd6 = nn.AdaptiveMaxPool2d((96,96))
        self.h3_PT_hd6_conv = nn.Conv2d(filters[2],120, 3, padding=1)
        self.h3_PT_hd6_bn = nn.BatchNorm2d(120)
        self.h3_PT_hd6_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.h5_UT_hd6 = nn.Upsample(scale_factor=3, mode='bilinear')  # 14*14
        self.h5_UT_hd6_conv = nn.Conv2d(filters[4], 120, 3, padding=1)
        self.h5_UT_hd6_bn = nn.BatchNorm2d(120)
        self.h5_UT_hd6_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd7_UT_hd6 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.hd7_UT_hd6_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd7_UT_hd6_bn = nn.BatchNorm2d(120)
        self.hd7_UT_hd6_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv6d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu6d_1 = nn.ReLU(inplace=True)

        '''stage 5d '''

        # h4->40*40, hd4->40*40, Concatenation
        self.h3_Cat_hd5_conv = nn.Conv2d(filters[2],90, 3, padding=1)
        self.h3_Cat_hd5_bn = nn.BatchNorm2d(90)
        self.h3_Cat_hd5_relu = nn.ReLU(inplace=True)

        # h3->256*256, hd6->96*96, Pooling 4 times
        self.h2_PT_hd5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd5_conv = nn.Conv2d(filters[2], 90, 3, padding=1)
        self.h2_PT_hd5_bn = nn.BatchNorm2d(90)
        self.h2_PT_hd5_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd7_UT_hd5 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd7_UT_hd5_conv = nn.Conv2d(360,90, 3, padding=1)
        self.hd7_UT_hd5_bn = nn.BatchNorm2d(90)
        self.hd7_UT_hd5_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd6_UT_hd5 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False) # 14*14
        self.hd6_UT_hd5_conv = nn.Conv2d(360, 90, 3, padding=1)
        self.hd6_UT_hd5_bn = nn.BatchNorm2d(90)
        self.hd6_UT_hd5_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv5d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu5d_1 = nn.ReLU(inplace=True)

        '''stage 4d'''
        # h3->256*256, hd6->96*96, Pooling 4 times
        self.h2_PT_hd4 = nn.AdaptiveMaxPool2d((192, 192))
        self.h2_PT_hd4_conv = nn.Conv2d(filters[2], 120, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(120)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd6_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd6_UT_hd4_conv = nn.Conv2d(filters[4], 120, 3, padding=1)
        self.hd6_UT_hd4_bn = nn.BatchNorm2d(120)
        self.hd6_UT_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd5_UT_hd4 =nn.Upsample(size=(192, 192), mode='bilinear', align_corners=False) # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(120)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''

        # h4->40*40, hd4->40*40, Concatenation
        self.h2_Cat_hd3_conv = nn.Conv2d(filters[2], 90, 3, padding=1)
        self.h2_Cat_hd3_bn = nn.BatchNorm2d(90)
        self.h2_Cat_hd3_relu = nn.ReLU(inplace=True)

        # h3->256*256, hd6->96*96, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], 90, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(90)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd4_UT_hd3 = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False) # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(360, 90, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(90)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(360, 90, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(90)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d'''
        # h3->256*256, hd6->96*96, Pooling 4 times
        self.h1_PT_hd2 = nn.AdaptiveMaxPool2d((384, 384))
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], 120, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(120)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd3_UT_hd2 =nn.Upsample(size=(384, 384), mode='bilinear', align_corners=False)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(120)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd4_UT_hd2 =nn.Upsample(scale_factor=2, mode='bilinear') # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(120)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h4->40*40, hd4->40*40, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], 120, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(180)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd2_UT_hd1 = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False) # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(120)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times对应上采样第5层->4层
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(360, 120, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(120)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------

        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        h5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        #for hd7-------------------------------------
        h3_PT_hd7 = self.h3_PT_hd7_relu(self.h3_PT_hd7_bn(self.h3_PT_hd7_conv(self.h3_PT_hd7(h3))))#把h3向下处理成hd7可用
        h4_Cat_hd7 = self.h4_Cat_hd7_relu(self.h4_Cat_hd7_bn(self.h4_Cat_hd7_conv(h4)))#把h4平行处理成hd7可用
        h5_UT_hd7 = self.h5_UT_hd7_relu(self.h5_UT_hd7_bn(self.h5_UT_hd7_conv(self.h5_UT_hd7(h5))))#把h5向上处理成hd7可用
        hd7 = self.relu7d_1(self.bn7d_1(self.conv7d_1(
            torch.cat((h3_PT_hd7,h4_Cat_hd7,h5_UT_hd7), 1))))  # hd4->40*40*UpChannels#拼接准备好的可用通道特征图

        # for hd6-------------------------------------
        h3_PT_hd6 = self.h3_PT_hd6_relu(self.h3_PT_hd6_bn(self.h3_PT_hd6_conv(self.h3_PT_hd6(h3))))  # 把h3向下处理成hd6可用
        h5_UT_hd6 = self.h5_UT_hd6_relu(self.h5_UT_hd6_bn(self.h5_UT_hd6_conv(self.h5_UT_hd6(h5))))# 把hd5向上处理成hd6可用
        hd7_UT_hd6 = self.hd7_UT_hd6_relu(self.hd7_UT_hd6_bn(self.hd7_UT_hd6_conv(self.hd7_UT_hd6(hd7))))
        hd6 = self.relu6d_1(self.bn6d_1(self.conv6d_1(
            torch.cat((h3_PT_hd6,h5_UT_hd6,hd7_UT_hd6), 1))))  # hd3->80*80*UpChannels

        # for hd5-------------------------------------
        h3_Cat_hd5 = self.h3_Cat_hd5_relu(self.h3_Cat_hd5_bn(self.h3_Cat_hd5_conv(h3)))
        h2_PT_hd5 = self.h2_PT_hd5_relu(self.h2_PT_hd5_bn(self.h2_PT_hd5_conv(self.h2_PT_hd5(h2))))  # 把h3向下处理成hd6可用
        hd7_UT_hd5 = self.hd7_UT_hd5_relu(self.hd7_UT_hd5_bn(self.hd7_UT_hd5_conv(self.hd7_UT_hd5(hd7))))
        hd6_UT_hd5 = self.hd6_UT_hd5_relu(self.hd6_UT_hd5_bn(self.hd6_UT_hd5_conv(self.hd6_UT_hd5(hd6))))
        hd5 = self.relu5d_1(self.bn5d_1(self.conv5d_1(
            torch.cat((h3_Cat_hd5, h2_PT_hd5, hd7_UT_hd5, hd6_UT_hd5), 1))))  # hd2->160*160*UpChannels

        # for hd4-------------------------------------
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))  # 把h3向下处理成hd6可用
        hd6_UT_hd4 = self.hd6_UT_hd4_relu(self.hd6_UT_hd4_bn(self.hd6_UT_hd4_conv(self.hd6_UT_hd4(hd6))))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h2_PT_hd4,hd6_UT_hd4,hd5_UT_hd4), 1))))  # hd1->320*320*UpChannels

        # for hd3-------------------------------------
        h2_Cat_hd3 = self.h2_Cat_hd3_relu(self.h2_Cat_hd3_bn(self.h2_Cat_hd3_conv(h2)))
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))  # 把h3向下处理成hd6可用
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h2_Cat_hd3,h1_PT_hd3,hd4_UT_hd3,hd5_UT_hd3), 1))))  # hd2->160*160*UpChannels

        # for hd2-------------------------------------
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))  # 把h3向下处理成hd6可用
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2 ,hd3_UT_hd2,hd4_UT_hd2), 1))))  # hd1->320*320*UpChannels

        # for hd1-------------------------------------
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1,hd2_UT_hd1,hd3_UT_hd1), 1))))  # hd1->320*320*UpChannels
        d1 = self.outconv1(hd1)  # d1->320*320*n_classes


        return d1


