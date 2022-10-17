#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2022/3/19 22:46
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description:


import numpy as np
import torch
from model.TFBS_ATT_model import TFBS_ATT
from tqdm import tqdm
from torch.cuda.amp import autocast # 用于混合精度运算
from osgeo import gdal,gdalconst

def pred_allModel(net,in_img,out_img,mixAcc=True,soft=False):
    stride = 128
    image_size = 128
    img_c = 2
    img_t = 9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = load_ConvLSTM_net(netpath=weightpath, is_net_structure=isNet, device=device)
    net.eval()
    # load the image
    inputdst = gdal.Open(in_img)
    if inputdst is None:
        print("can't open " + in_img)
        return False
    # get the metadata of data
    w = inputdst.RasterXSize  # 原始栅格矩阵的列数
    h = inputdst.RasterYSize  # 原始栅格矩阵的行数
    c = inputdst.RasterCount  # 原始栅格矩阵的波段数
    X_prj = inputdst.GetProjection()  # 获取投影类型
    adfGeoTransform = inputdst.GetGeoTransform()  # 获取原始数据的转换7参数

    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    # label输出
    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float)

    for i in tqdm(range(padding_h // stride)):
        for j in range(padding_w // stride):
            crop = np.zeros((c, image_size, image_size), dtype=np.int16)
            if j == padding_w // stride - 1:
                if i == padding_h // stride - 1:
                    crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                    ysize=h % stride)
                    crop[:, 0:h % stride, 0:w % stride] = crop_tmp
                else:
                    crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                    ysize=image_size)
                    crop[:, :, 0:w % stride] = crop_tmp
            else:
                if i == padding_h // stride - 1:
                    crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=image_size,
                                                    ysize=h % stride)
                    crop[:, 0:h % stride, :] = crop_tmp
                else:
                    crop = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=image_size,
                                                ysize=image_size)

            cb, ch, cw = crop.shape
            if ch != image_size or cw != image_size:
                print('invalid size!')
                return

            # 做归一化处理
            min_crop = np.min(crop)
            if min_crop == 0:
                continue
            else:
                # 做归一化处理
                crop = -crop / 4000.0

                crop = np.expand_dims(crop, axis=0)
                crop_tensor = torch.from_numpy(crop)

                # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
                if mixAcc:
                    # 将数据拷贝到device中
                    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
                    image = crop_tensor.to(device=device, dtype=torch.bfloat16)
                    with autocast():
                        # 使用网络参数，输出预测结果
                        pred = net(image)
                        pred = np.array(pred.data.cpu()[0])[0]
                        # 处理结果
                        pred[pred >= 0] = 1
                        pred[pred < 0] = 0
                        mask_whole[i * stride:i * stride + image_size, j *
                                                                       stride:j * stride + image_size] = pred[:, :]
                else:
                    image = crop_tensor.to(device=device, dtype=torch.float32)
                    # 使用网络参数，输出预测结果
                    pred = net(image)
                    pred = np.array(pred.data.cpu()[0])[0]
                    # 处理结果
                    pred[pred >= 0] = 1
                    pred[pred < 0] = 0
                    # print 'pred:',pred.shape
                    mask_whole[i * stride:i * stride + image_size, j *
                                                                   stride:j * stride + image_size] = pred[:, :]

    result = mask_whole[0:h, 0:w]
    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式

    if not soft:
        result = np.round(result).astype(np.uint8)
        rstlabelDst = tifDriver.Create(out_img, w, h, 1,
                                       gdalconst.GDT_Byte);  # 创建目标文件

    else:
        isByte = True #默认节省空间
        if isByte:
            result = np.rint(result * 100).astype(np.uint8)  # 减少数据量，投射到整型概率
            rstlabelDst = tifDriver.Create(out_img, w, h, 1,
                                           gdalconst.GDT_Byte);  # 创建目标文件
        else:
            rstlabelDst = tifDriver.Create(out_img, w, h, 1,
                                           gdalconst.GDT_Float32);  # 创建目标文件

    rstlabelDst.SetGeoTransform(adfGeoTransform)  # 写入仿射变换参数
    rstlabelDst.SetProjection(X_prj)  # 写入投影
    rstlabelDst.GetRasterBand(1).WriteArray(result)  # 写入label数据

def load_attTFBS_net(netpath,is_net_structure,device):
    if is_net_structure:
        return torch.load(netpath, map_location=device)
    else:
        net = TFBS_ATT(bs=16, n_laryers=1, hiddensize=64)
        net.to(device=device)
        net.load_state_dict(torch.load(netpath, map_location=device))
        return net

def pred_allModel_Main():
    pthpath = 'F:/PTH/attTFBS_20220715-140352-_best_val_loss_weight.pth'
    in_img = r'D:\2019\ID03_2019-04-01To2019-11-02_S1_VVVH_10m_MEDIAN.tif'
    out_img = r'F:\03-classification\2019\ID03-2019-20220715-140352.tif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_attTFBS_net(netpath=pthpath, is_net_structure=False, device=device)
    pred_allModel(net, in_img, out_img, mixAcc=True, soft=False)
    print('done!')


if __name__ == "__main__":
    pred_allModel_Main()