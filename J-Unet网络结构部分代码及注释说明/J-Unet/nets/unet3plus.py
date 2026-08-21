# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers import unetConv2
from nets.init_weights import init_weights
#基于Unet+++修改后的Unet模型
'''
    UNet 3+
'''

#残差
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        """
        该类定义了一个具有跳过连接的卷积块。用于构建残差网络。

        :param input_dim: 输入特征的数量
        :param output_dim: 输出特征的数量
        :param stride: 卷积步长
        :param padding: 卷积填充
        """
        super(ResidualConv, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
    def forward(self, x):
        """
        定义前向传播过程。

        :param x: 输入到此卷积块的数据
        :return: 经过卷积层和批量归一化层处理的数据
        """
        return self.conv_skip(x)

class UNet_3Plus(nn.Module):

    def __init__(self, n_classes):
        super(UNet_3Plus, self).__init__()
        # self.args = args
        in_channels = 3 #设置输入图像的通道数为3 即RGB三个通道
        n_classes = n_classes #分类数设置为3 （背景 锥体 椎间盘）
        feature_scale = 4 #原始代码中用到的，修改后已经弃用
        is_deconv = True #原始代码中用到的，修改后已经弃用
        is_batchnorm = True #决定是否在卷积网络中使用Batch Normalization（批量归一化
        self.is_deconv = is_deconv #原始代码中用到的，修改后已经弃用
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]#修改后只是大部分只是为了取用该变量当中的值

        ## -------------Encoder--------------下采样编码阶段的预先准备（后面要调用这些方法）
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)#(1 3 512 512)->(1 64 512 512)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)#最大池化，缩小特征图tensor(1 64 512 512)->(1 64 256 256)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)  # （1 64 256 256）->（1 128 256 256）
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)#（1 128 256 256）-> (1 128 128 128)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)  # （1 128 128 128）->(1 256 128 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)#(1 256 128 128)->(1 256 64 64)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)#(1 256 64 64)->(1 512 64 64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)#(1 512 64 64)->(1 512 32 32)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)#(1 1024 32 32)

        ## -------------Decoder--------------
        #上采样阶段的预先准备
        self.CatChannels = filters[0]#已经弃用
        self.CatBlocks = 5#已经弃用
        self.UpChannels = 180 #自定义了该值
        self.res7_6=ResidualConv(180,180, 1, 1)#预定义残差块，方便进行残差连接
        self.res6_5 = ResidualConv(180, 180, 1, 1)
        self.res5_4 = ResidualConv(180, 180, 1, 1)
        self.res4_3 = ResidualConv(180, 180, 1, 1)
        self.res3_2 = ResidualConv(180, 180, 1, 1)
        self.res2_1 = ResidualConv(180, 180, 1, 1)
        #参考网络结构图左边五个块从上倒下依次记为【h1~h5】(包括并将最下面中间的块记作h5)
        # 右边七个块从上到下依次记为【hd1~hd7】
        #【PT】向下变形（变小）例如（1 64 128 128）->（1 64 256 256）且通道数不变
        #【UT】向上变形（变大）例如...
        #【Cat】无变形（可直接拼接）
        # 【h?】_【PT/UT/CAT】_【hd?】代表前者变形成后者，以用于拼接，例如：h3_PT_hd7,说明h3变形成hd7的形状
        '''stage 7d'''#为hd7面临的三个指向它的箭头对应的处理而写的代码 对应卷积操作，激活操作，池化操作
        # h3->128*128, hd7->64*64, Pooling 2 times
        self.h3_PT_hd7 = nn.MaxPool2d(2, 2, ceil_mode=True)#池化操作，具体数值变化可自行搜索
        self.h3_PT_hd7_conv = nn.Conv2d(filters[2], 60, 3, padding=1)
        self.h3_PT_hd7_bn = nn.BatchNorm2d(60)
        self.h3_PT_hd7_relu = nn.ReLU(inplace=True)


        # h4->64*64, hd7->64*64, Concatenation 二者长宽一样，可直接拼接
        self.h4_Cat_hd7_conv = nn.Conv2d(filters[3],60, 3, padding=1)
        self.h4_Cat_hd7_bn = nn.BatchNorm2d(60)
        self.h4_Cat_hd7_relu = nn.ReLU(inplace=True)


        # h5->32*32, hd7->64*64, Upsample 2 times对应h5层->hd7层的形变
        self.h5_UT_hd7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.h5_UT_hd7_conv = nn.Conv2d(filters[4], 60, 3, padding=1)
        self.h5_UT_hd7_bn = nn.BatchNorm2d(60)
        self.h5_UT_hd7_relu = nn.ReLU(inplace=True)

        #针对hd7融合部分的相关操作准备，原本的unet+++也有这些操作，为什么需要在拼接之后进行这些操作需要进一步了解
        self.conv7d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn7d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu7d_1 = nn.ReLU(inplace=True)

        '''stage 6d'''
        # h3->128*128, hd6->96*96, AdaptiveMaxPool2d#使用自适应最大池化用于变形成适合拼接的形状
        self.h3_PT_hd6 = nn.AdaptiveMaxPool2d((96,96))
        self.h3_PT_hd6_conv = nn.Conv2d(filters[2],60, 3, padding=1)
        self.h3_PT_hd6_bn = nn.BatchNorm2d(60)
        self.h3_PT_hd6_relu = nn.ReLU(inplace=True)

        # h5->32*32, hd6->96*96, Upsample 3 times对应h5->hd6的形状变化
        self.h5_UT_hd6 = nn.Upsample(scale_factor=3, mode='bilinear')  # 14*14
        self.h5_UT_hd6_conv = nn.Conv2d(filters[4], 60, 3, padding=1)
        self.h5_UT_hd6_bn = nn.BatchNorm2d(60)
        self.h5_UT_hd6_relu = nn.ReLU(inplace=True)

        # hd7->64*64, hd6->96*96, 对应上采样第5层->4层
        #采用双线性插值(bilinear)方式进行上采样，以变形至可拼接的形状
        self.hd7_UT_hd6 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.hd7_UT_hd6_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd7_UT_hd6_bn = nn.BatchNorm2d(60)
        self.hd7_UT_hd6_relu = nn.ReLU(inplace=True)

        # 针对hd6融合部分的相关操作准备，原本的unet+++也有这些操作，为什么需要在拼接之后进行这些操作需要进一步了解
        self.conv6d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu6d_1 = nn.ReLU(inplace=True)

        '''stage 5d '''

        # h3->128*128, hd5->128*128, Concatenation
        self.h3_Cat_hd5_conv = nn.Conv2d(filters[2],45, 3, padding=1)
        self.h3_Cat_hd5_bn = nn.BatchNorm2d(45)
        self.h3_Cat_hd5_relu = nn.ReLU(inplace=True)

        # h2->256*256, hd5->128*128, Pooling 2 times
        self.h2_PT_hd5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd5_conv = nn.Conv2d(filters[1], 45, 3, padding=1)
        self.h2_PT_hd5_bn = nn.BatchNorm2d(45)
        self.h2_PT_hd5_relu = nn.ReLU(inplace=True)

        # hd7->64*64, hd5->128*128, Upsample 2 times对应上采样第5层->4层
        self.hd7_UT_hd5 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd7_UT_hd5_conv = nn.Conv2d(180,45, 3, padding=1)
        self.hd7_UT_hd5_bn = nn.BatchNorm2d(45)
        self.hd7_UT_hd5_relu = nn.ReLU(inplace=True)

        # hd6->96*96, hd5->128*128
        # 采用双线性插值(bilinear)方式进行上采样，以变形至可拼接的形状
        self.hd6_UT_hd5 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False) # 14*14
        self.hd6_UT_hd5_conv = nn.Conv2d(180, 45, 3, padding=1)
        self.hd6_UT_hd5_bn = nn.BatchNorm2d(45)
        self.hd6_UT_hd5_relu = nn.ReLU(inplace=True)

        # 针对hd5融合部分的相关操作准备，原本的unet+++也有这些操作，为什么需要在拼接之后进行这些操作需要进一步了解
        self.conv5d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu5d_1 = nn.ReLU(inplace=True)

        #同理 4d 3d 2d 1d的操作与前部类似
        '''stage 4d'''
        # h2->256*256, hd4->192*192, Pooling 4 times
        self.h2_PT_hd4 = nn.AdaptiveMaxPool2d((192, 192))
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], 60, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(60)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # hd6->96*96, hd4->192*192, Upsample 2 times对应上采样第5层->4层
        self.hd6_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd6_UT_hd4_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd6_UT_hd4_bn = nn.BatchNorm2d(60)
        self.hd6_UT_hd4_relu = nn.ReLU(inplace=True)

        # hd5->128*128, hd4->192*192
        self.hd5_UT_hd4 =nn.Upsample(size=(192, 192), mode='bilinear', align_corners=False) # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(60)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)


        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        #请参考上方注释（7d / 6d），思路方法一致
        '''stage 3d'''

        # h2->256*256, hd3->256*256, Concatenation
        self.h2_Cat_hd3_conv = nn.Conv2d(filters[1], 45, 3, padding=1)
        self.h2_Cat_hd3_bn = nn.BatchNorm2d(45)
        self.h2_Cat_hd3_relu = nn.ReLU(inplace=True)

        # h1->512*512, hd3->256*256, Pooling 2 times
        self.h1_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], 45, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(45)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # hd4->192*192, hd3->256*256
        self.hd4_UT_hd3 = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False) # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(180, 45, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(45)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->128*128, hd3->256*256, Upsample 2 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(180, 45, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(45)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)


        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d'''
        # h1->512*512, hd2->384*384
        self.h1_PT_hd2 = nn.AdaptiveMaxPool2d((384, 384))
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], 60, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(60)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # hd3->256*256, hd2->384*384
        self.hd3_UT_hd2 =nn.Upsample(size=(384, 384), mode='bilinear', align_corners=False)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(60)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->192*192, hd2->384*384
        self.hd4_UT_hd2 =nn.Upsample(scale_factor=2, mode='bilinear') # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(60)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->512*512, hd1->512*512, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], 60, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(60)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->384*384, hd4->192*192
        self.hd2_UT_hd1 = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False) # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(60)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->256*256, hd1->512*512, Upsample 2 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(180, 60, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(60)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)


        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output#对应输出的特征图（1 3 512 512）
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        """这段代码中，for m in self.modules():循环遍历了模型中的所有模块。
        如果某个模块是nn.Conv2d（二维卷积层）或者nn.BatchNorm2d（二维批量归一化层），
        就使用init_weights函数来初始化这个模块的权重。
        init_weights(m, init_type='kaiming')这一行代码的具体含义取决于init_weights函数的实现。
        根据函数参数init_type='kaiming'进行Kaiming初始化（也被称为He初始化），
        这是一种专门用于ReLU激活函数及其变体的权重初始化方法。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        #前向传播的下采样部分
        h1 = self.conv1(inputs)  # h1->（1 64 512 512）

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->（1 128 256 256）

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->（1 256 128 128）

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->（1 512 64 64）

        h5 = self.maxpool4(h4)
        h5 = self.conv5(h5)  # h5->（1 1024 32 32）

        ## -------------Decoder-------------
        #上采样部分
        #for hd7-------------------------------------
        #首先准备好待拼接的四个tensor,相当于网络结构图中的”箭头“
        #通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[h3_PT_hd7 h4_Cat_hd7 h5_UT_hd7]
        #此时他们三个的形状都是【1 60 64 64】拼接后变成hd7【1 180 64 64】
        h3_PT_hd7 = self.h3_PT_hd7_relu(self.h3_PT_hd7_bn(self.h3_PT_hd7_conv(self.h3_PT_hd7(h3))))
        h4_Cat_hd7 = self.h4_Cat_hd7_relu(self.h4_Cat_hd7_bn(self.h4_Cat_hd7_conv(h4)))
        h5_UT_hd7 = self.h5_UT_hd7_relu(self.h5_UT_hd7_bn(self.h5_UT_hd7_conv(self.h5_UT_hd7(h5))))
        hd7 = self.relu7d_1(self.bn7d_1(self.conv7d_1(
            torch.cat((h3_PT_hd7,h4_Cat_hd7,h5_UT_hd7), 1))))
        #temp用于缓存残差连接要用的残差块，后续要经过一步激活处理，这一步的原因需参考相关文献
        temp=self.res7_6(hd7)+hd7
        hd7=self.relu7d_1(temp)

        # for hd6-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[h3_PT_hd6 h5_UT_hd6 hd7_UT_hd6]
        # 此时他们三个的形状都是【1 60 96 96】拼接后变成hd6【1 180 96 96】因为3*60=180
        h3_PT_hd6 = self.h3_PT_hd6_relu(self.h3_PT_hd6_bn(self.h3_PT_hd6_conv(self.h3_PT_hd6(h3))))  # 把h3向下处理成hd6可用
        h5_UT_hd6 = self.h5_UT_hd6_relu(self.h5_UT_hd6_bn(self.h5_UT_hd6_conv(self.h5_UT_hd6(h5))))# 把hd5向上处理成hd6可用
        hd7_UT_hd6 = self.hd7_UT_hd6_relu(self.hd7_UT_hd6_bn(self.hd7_UT_hd6_conv(self.hd7_UT_hd6(hd7))))
        hd6 = self.relu6d_1(self.bn6d_1(self.conv6d_1(
            torch.cat((h3_PT_hd6,h5_UT_hd6,hd7_UT_hd6), 1))))  # hd3->80*80*UpChannels
        temp = self.res6_5(hd6) + hd6
        hd6 = self.relu6d_1(temp)


        # for hd5-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[h3_Cat_hd5 h2_PT_hd5 hd7_UT_hd5 hd6_UT_hd5]
        # 此时他们四个的形状都是【1 45 96 96】拼接后变成hd5【1 180 128 128】 因为45*4=180
        h3_Cat_hd5 = self.h3_Cat_hd5_relu(self.h3_Cat_hd5_bn(self.h3_Cat_hd5_conv(h3)))
        h2_PT_hd5 = self.h2_PT_hd5_relu(self.h2_PT_hd5_bn(self.h2_PT_hd5_conv(self.h2_PT_hd5(h2))))  # 把h3向下处理成hd6可用
        hd7_UT_hd5 = self.hd7_UT_hd5_relu(self.hd7_UT_hd5_bn(self.hd7_UT_hd5_conv(self.hd7_UT_hd5(hd7))))
        hd6_UT_hd5 = self.hd6_UT_hd5_relu(self.hd6_UT_hd5_bn(self.hd6_UT_hd5_conv(self.hd6_UT_hd5(hd6))))
        hd5 = self.relu5d_1(self.bn5d_1(self.conv5d_1(
            torch.cat((h3_Cat_hd5, h2_PT_hd5, hd7_UT_hd5, hd6_UT_hd5), 1))))  # hd2->160*160*UpChannels
        temp = self.res5_4(hd5) + hd5
        hd5 = self.relu5d_1(temp)

        # for hd4-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[ h2_PT_hd4 hd6_UT_hd4 hd5_UT_hd4]
        # 此时他们三个的形状都是【1 60 128 128】拼接后变成hd4【1 180 192 192】
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))  # 把h3向下处理成hd6可用
        hd6_UT_hd4 = self.hd6_UT_hd4_relu(self.hd6_UT_hd4_bn(self.hd6_UT_hd4_conv(self.hd6_UT_hd4(hd6))))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h2_PT_hd4,hd6_UT_hd4,hd5_UT_hd4), 1))))  # hd1->320*320*UpChannels
        temp = self.res4_3(hd4) + hd4
        hd4 = self.relu4d_1(temp)

        # for hd3-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[ h2_PT_hd4 hd6_UT_hd4 hd5_UT_hd4]
        # 此时他们四个的形状都是【1 45 192 192】拼接后变成hd3【1 180 256 256】
        h2_Cat_hd3 = self.h2_Cat_hd3_relu(self.h2_Cat_hd3_bn(self.h2_Cat_hd3_conv(h2)))
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))  # 把h3向下处理成hd6可用
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h2_Cat_hd3,h1_PT_hd3,hd4_UT_hd3,hd5_UT_hd3), 1))))  # hd2->160*160*UpChannels
        temp = self.res3_2(hd3) + hd3
        hd3 = self.relu3d_1(temp)

        # for hd2-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[h1_PT_hd2 hd3_UT_hd2 hd4_UT_hd2]
        # 此时他们三个的形状都是【1 60 256 256】拼接后变成hd2【1 180 384 384】
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))  # 把h3向下处理成hd6可用
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2 ,hd3_UT_hd2,hd4_UT_hd2), 1))))  # hd1->320*320*UpChannels
        temp = self.res2_1(hd2) + hd2
        hd2 = self.relu2d_1(temp)

        # for hd1-------------------------------------
        # 通过事先封装定义的操作完成这些操作，得到了可以被用于torch.cat()的[h1_Cat_hd1 hd2_UT_hd1 hd3_UT_hd1]
        # 此时他们三个的形状都是【1 60 384 384】拼接后变成hd1【1 180 512 512】
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1,hd2_UT_hd1,hd3_UT_hd1), 1))))
        #d1是要返回的最终的特征图，它后续将经过train.py中的softmax操作得出预测值
        d1 = self.outconv1(hd1)  # d1->（1 3 512 512）


        return d1


