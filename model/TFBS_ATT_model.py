#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2022/4/22 10:15
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description:


import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.Upsample(scale_factor=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #logits = self.outc(x)
        return x#logits

class UNet32(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet32, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.up4 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class TFBS_ATT(nn.Module):
    def __init__(self, n_features=2, hiddensize=64, n_laryers=1, seq_len=9, bs=16,bilinear=True):
        '''
        LSTM model init
        :param n_features: the number of the features
        :param hiddensize: the size of the hidden features
        :param n_laryers:  the number of the hidden layers
        :param seq_len: the length of each input seq
        :param batchsize: batch size
        '''
        super(TFBS_ATT, self).__init__()
        self.n_features = n_features
        self.hiddensize = hiddensize
        self.n_laryers = n_laryers
        self.seq_len = seq_len
        self.bs = bs
        self.bilinear=bilinear

        self.lstm = nn.LSTM(n_features,hiddensize,n_laryers,batch_first=True) # batch first

        #self.unet = UNet32(n_channels=hiddensize, n_classes=1,bilinear=bilinear)
        self.unet = UNet(n_channels=hiddensize, bilinear=bilinear)
        self.outc = OutConv(hiddensize, out_channels=1)

    def forward(self, x, intermediate_fea=False): #input shape of x: [bs,c, h, w] [bs, 18, 128, 128]
        # 1st： 转变为 [bs, h, w, c] [bs, 128, 128 ,18]
        x = x.transpose(1, 2)  # [bs, h, c, w]
        x = x.transpose(2, 3)  # [bs, h, w, c]
        # 2nd： 转变为 [bs*h*w, c] [bs*h*w, 18] => 转变为 [bs*h*w, n_feature, seq_len] [bs*h*w, 2, 9]
        #x = x.contiguous().view(self.bs*128*128, 2, 9) #这个128是图像的长和宽
        x = x.contiguous().view(-1, 2, 9) #这个128是图像的长和宽
        # 3rd： 转变为 [bs*h*w, 9, 2]
        x = x.transpose(1, 2)
        #the input shape should be : [batchsize, 9, 2] which are bs, n_feature, hiddensize, respectively.
        # x1的维度[bs_pixels,9,64], h_n的维度[1,bs_p,64], c_n的维度[1,bs_p,64]
        x1, (h_n, c_n) = self.lstm(x) # shape of h_n: [ n_layers, batchsize, hiddensize] = [1,16384, 128]
        '''h_n = h_n.transpose(0, 1) # [16384, 1, 64]
        # shape of out: [batchsize, n_layers, hiddensize] [bs, 1, 64] => [bs, 1, 128] => [bs, 64]
        h_n = h_n[:, self.n_laryers-1, :].view(-1, self.hiddensize)'''
        #x1 = x1[:, -1, :] #[bs, 64]
        #注意力机制
        H = torch.nn.Tanh()(x1) # 将数据用tanh激活函数压缩到-1和1之间,维度[bs_pixels,9,64]
        bs_true = x1.shape[0] #LSTM输出数据的维度
        w_omiga = torch.randn(bs_true, self.hiddensize, 1, requires_grad=True).cuda() # [bs_pixels, 64, 1]
        #weights的维度是[bs_pixels, 9, 64]
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1, 1,
                                                                                                     self.hiddensize)
        attention_output = torch.mul(x1, weights).sum(dim=-2)  # [batch_size,hidden_dim]
        # 将输出的二维向量，重组织为三维的图像 =>[bs, h, w, hiddensize]
        #f_i = h_n.view(self.bs, 128, 128, self.hiddensize)
        f_i = attention_output.view(-1, 128, 128, self.hiddensize)
        # 将特征数量维度调换到第二的位置 => [bs, hiddensize, h, w]
        f_i = f_i.permute(0, 3, 1, 2)
        # 输入到unet结构中 => [bs, 1, h, w]
        out = self.unet(f_i)
        if not intermediate_fea:
            out = self.outc(out)
        return out

if __name__ == '__main__':
    net = TFBS_ATT()
    print(net)