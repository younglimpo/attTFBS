#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2022/3/19 22:16
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description:

import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import random
from osgeo import gdal,gdalconst

class ISBI_Loader(Dataset):
    def __init__(self, data_path,isFile = False):
        # 初始化函数，读取所有data_path下的图片
        if isFile:#如果输入的是一连串的file path的list
            self.imgs_path = data_path
        else:
            self.data_path = data_path
            self.imgs_path = glob.glob(os.path.join(data_path, 'src/*.tiff'))
            if os.path.exists(os.path.join(data_path,'src/augmentation')):
                self.imgs_path.extend(glob.glob(os.path.join(data_path, 'src/augmentation/*.tiff')))


    def load_img(self, path, scaled=True):
        inputdst = gdal.Open(path)
        if inputdst is None:
            print("can't open " + path)
            return False
        # get the metadata of data
        X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
        X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
        src_img = inputdst.ReadAsArray(
                xoff=0, yoff=0, xsize=X_width, ysize=X_height)  # [c, h, w] c = 18对于研究区
        if scaled == True:
                src_img = (-np.array(src_img, dtype="float32")) / 4000.
        return src_img

    def load_label(self, path, GrayScaled=True):
        inputdst = gdal.Open(path)
        if inputdst is None:
            print("can't open " + path)
            return False
        # get the metadata of data
        X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
        X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height)
        if GrayScaled:
            src_img = src_img.reshape(1, X_height, X_width)
        return src_img

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('src', 'label')
        # 读取训练图片和标签图片
        image = self.load_img(image_path)
        label = self.load_label(label_path)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

class ISBI_Loader_CV(Dataset):
    def __init__(self, img_path):
        # 初始化函数，读取所有data_path下的图片
        self.imgs_path = img_path

    def load_img(self, path, scaled=True):
        inputdst = gdal.Open(path)
        if inputdst is None:
            print("can't open " + path)
            return False
        # get the metadata of data
        X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
        X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
        src_img = inputdst.ReadAsArray(
                xoff=0, yoff=0, xsize=X_width, ysize=X_height)  # [c, h, w] c = 18对于研究区
        if scaled == True:
                src_img = (-np.array(src_img, dtype="float32")) / 4000.
        return src_img

    def load_label(self, path, GrayScaled=True):
        inputdst = gdal.Open(path)
        if inputdst is None:
            print("can't open " + path)
            return False
        # get the metadata of data
        X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
        X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
        src_img = inputdst.ReadAsArray(
            xoff=0, yoff=0, xsize=X_width, ysize=X_height)
        if GrayScaled:
            src_img = src_img.reshape(1, X_height, X_width)
        return src_img

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('src', 'label')
        # 读取训练图片和标签图片
        image = self.load_img(image_path)
        label = self.load_label(label_path)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

if __name__ == "__main__":
    datapath = ''
    isbi_dataset = ISBI_Loader("E:/15-ARMSMOTN/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)