#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2020/11/06 15:23
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description: Generate train and test data
# coding=utf-8

import random
import os,shutil
import numpy as np
from tqdm import tqdm
from osgeo import gdal,gdalconst
import glob
img_w = 128
img_h = 128
junkID = '16'
head_name = 'ns'+junkID+'-'
head_name_seg = '2017ns'+junkID+'-'
head_name_seg = '2017SC'
#img_path = 'D:/09-NE-RiceMapping/09-samples/OneGoodModel/02-AllNegSamples-roi/NegSample' + junk_v + '/'
#image_sets = ['NS' + junk_v + '-src.tif'] # input source image
#label_sets = ['NS' + junk_v + '-label.tif'] # corresponding label image

#img_path = 'D:/09-NE-RiceMapping/09-samples/OneGoodModel/07-2021samples/02-allnegsamples/ns'+junkID+'/'
#image_sets = [head_name + 'src.tif'] # input source image
#label_sets = [head_name + 'label.tif'] # corresponding label image

img_path = 'F:/01-NorthEast/03-SC/2017/'
image_sets = ['2017-SC-VV-VH2.dat'] # input source image
label_sets = ['2017-SC-S2-rice-classification.dat'] # corresponding label image

label_foldername = 'label/'
def load_img(path):
    inputdst = gdal.Open(path)
    if inputdst is None:
        print("can't open " + inputdst)
        return False

    # get the metadata of data
    X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
    X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数

    src_img = inputdst.ReadAsArray(
        xoff=0, yoff=0, xsize=X_width, ysize=X_height)
    return src_img




def rotate_90(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    return xb, yb

def rotate_180(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    return xb, yb

def rotate_270(xb, yb):
    xb = np.transpose(xb,axes=(0,2,1)) #np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb,2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)
    xb = np.transpose(xb, axes=(0, 2, 1))  # np.transpose(xb,axes=(0,2,1))
    xb = np.flip(xb, 2)

    yb = np.transpose(yb)
    yb = np.flip(yb,1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    yb = np.transpose(yb)
    yb = np.flip(yb, 1)
    return xb, yb



def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def add_noise2(img):
    for i in range(int(img.shape[1]*img.shape[2]/50)):  # 添加点噪声
        for ii in range(img.shape[0]):
            temp_x = np.random.randint(0, img.shape[1])
            temp_y = np.random.randint(0, img.shape[2])
            img[ii,temp_x,temp_y] = np.random.randint(-3000, -1500)
    return img


def data_augment2(xb, yb): # only rotate and flip

    if np.random.random() < 0.25:
        xb, yb = rotate_90(xb, yb)
    if np.random.random() < 0.25:
        xb, yb = rotate_180(xb, yb)
    if np.random.random() < 0.25:
        xb, yb = rotate_270(xb, yb)
    if np.random.random() < 0.25:
        xb = np.flip(xb, 1)  # flipcode = 1：沿x轴翻转
        yb = np.flip(yb, 0)
    if np.random.random() < 0.25:
        xb = np.flip(xb, 2)  # flipcode = 2：沿y轴翻转
        yb = np.flip(yb, 1)
    if np.random.random() < 0.618:
        xb = add_noise2(xb) # add noise
    return xb, yb

def saveArray2disk(x,y,filename):
    """
    #将np数组保存到文件夹中
    :param x: train_x np数组 [c,h,w]
    :param y: 标签 y np数组 [h,w]
    :param filename:文件名 str，如 '1','2'
    :return: 文件名
    """
    c = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
    rstDst = tifDriver.Create(filename, w, h, c, gdalconst.GDT_Int16);  # 创建目标文件
    for ii in range(c):
        rstDst.GetRasterBand(ii + 1).WriteArray(x[ii])  # 写入数据
    rstlabelDst = tifDriver.Create((filename.replace('src','label')), w, h, 1,
                                   gdalconst.GDT_Byte);  # 创建目标文件
    rstlabelDst.GetRasterBand(1).WriteArray(y)  # 写入label数据
    return (filename)

def sample_augment(datapath,threshold=1000):
    """
    扩增正样本标签
    :param data: 数据列表list ['1',[2]...]
    :param threshold: 正样本像素点数量超过多少时会扩增
    :return: 扩增后的包含原样本的扩增样本
    """
    # print 'generateData...'
    data = glob.glob(os.path.join(datapath, 'src/*.tiff'))
    Augmentated_data = []
    # 删除原来的临时增广文件，创建新的增广文件夹
    if os.path.exists(os.path.join(datapath, 'src/augmentation')):
        shutil.rmtree(os.path.join(datapath, 'src/augmentation'))
    if os.path.exists(os.path.join(datapath, 'label/augmentation')):
        shutil.rmtree(os.path.join(datapath, 'label/augmentation'))
    os.mkdir(os.path.join(datapath, 'src/augmentation'))
    os.mkdir(os.path.join(datapath, 'label/augmentation'))

    g_count=0 #增广文件名，即1,2,3,4，等
    for i in (range(len(data))):
        print(i)
        url = data[i]
        Augmentated_data.append(url)
        # [band, row, column]
        train = load_img(url)
        # [row, column]
        label = load_img(url.replace('src', 'label'))

        if np.sum(label) < threshold: #当图片中正样本数量超过400个时扩增
            continue
        else:
            rdnum = np.random.randint(0, 3) #随机生成0-4的整数
            # 旋转90度
            if rdnum == 0:
                xb, yb = rotate_90(train, label)
                #xb = add_noise3(xb)  # add noise

            # 旋转180度
            if rdnum == 1:
                xb, yb = rotate_180(train, label)
                #xb = add_noise3(xb)  # add noise

            # 旋转270度
            if rdnum == 2:
                xb, yb = rotate_270(train, label)
                #xb = add_noise3(xb)  # add noise

            rdscr2 = np.random.randint(-100, 101) / 1000.0 + 1.0  # 随机生成0-4的整数
            xb = np.int16(xb * rdscr2)
            Augmentated_data.append(saveArray2disk(xb, yb, url.replace('src', 'src/augmentation')))
            g_count = g_count + 1

            rdnum = np.random.randint(0, 2)  # 随机生成0-4的整数
            if rdnum == 0:
                # 沿x轴翻转对称
                xb = np.flip(train, 1)  # flipcode = 1：沿x轴翻转
                yb = np.flip(label, 0)
                #xb = add_noise3(xb)  # add noise

            if rdnum == 1:
                # 沿y轴翻转对称
                xb = np.flip(train, 2)  # flipcode = 1：沿x轴翻转
                yb = np.flip(label, 1)
                #xb = add_noise3(xb)  # add noise

            rdscr2 = np.random.randint(-100, 101) / 1000.0 + 1.0  # 随机生成0-4的整数
            xb = np.int16(xb * rdscr2)
            Augmentated_data.append(saveArray2disk(xb, yb, url.replace('src\\', 'src\\augmentation\\f')))
            g_count = g_count + 1

            print('已扩增%d' % g_count)
    return Augmentated_data

def creat_dataset_random(image_num=100,Augment=True):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in range(len(image_sets)):
        count = 0
        # read input image and label data
        input_image=img_path  + image_sets[i]
        Srcdst = gdal.Open(input_image)
        if Srcdst == None:
            print("can't open " + Srcdst)
            return False
        input_label = img_path + label_sets[i]
        Labeldst = gdal.Open(input_label)
        if Labeldst == None:
            print("can't open " + Labeldst)
            return False

        # get the metadata of data
        X_width = Srcdst.RasterXSize  # 原始栅格矩阵的列数
        X_height = Srcdst.RasterYSize  # 原始栅格矩阵的行数
        X_bands = Srcdst.RasterCount  # 原始波段数
        X_eDT = Srcdst.GetRasterBand(1).DataType;  # 数据的类型
        X_prj = Srcdst.GetProjection()  # 获取投影类型
        adfGeoTransform = Srcdst.GetGeoTransform()  # 获取原始数据的转换7参数

        for count in tqdm(range(int(image_each))):
            random_width = random.randint(0, X_width - img_w - 1) # x_offset
            random_height = random.randint(0, X_height - img_h - 1) # y_offset
            src_roi = Srcdst.ReadAsArray(xoff=random_width,yoff=random_height,xsize=img_w,ysize=img_h)
            label_roi = np.byte(Labeldst.ReadAsArray(xoff=random_width,yoff=random_height,xsize=img_w,ysize=img_h))
            if Augment==True:
                src_roi,label_roi=data_augment2(src_roi,label_roi)
            tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
            rstDst = tifDriver.Create((img_path + 'src/%d.tiff' % g_count), img_w, img_h, X_bands, X_eDT);  # 创建目标文件
            for ii in range(X_bands):
                rstDst.GetRasterBand(ii + 1).WriteArray(src_roi[ii])  # 写入数据
            rstlabelDst = tifDriver.Create((img_path + label_foldername +'%d.tiff' % g_count), img_w, img_h, 1, X_eDT);  # 创建目标文件
            rstlabelDst.GetRasterBand(1).WriteArray(label_roi)  # 写入label数据
            count += 1
            g_count += 1

def creat_dataset_regular(stridevalue=128,image_size=128,Augment=False,headname = ''):
    # load the trained convolutional neural network
    stride = stridevalue  # args['stride']
    image_size = image_size
    print('creating dataset...')
    g_count = 0

    datapath = img_path + label_foldername

    # 删除原来的临时增广文件，创建新的增广文件夹
    if os.path.exists(img_path + label_foldername):
        shutil.rmtree(img_path + label_foldername)
    if os.path.exists(img_path + 'src/'):
        shutil.rmtree(img_path + 'src/')
    os.mkdir(img_path + 'src/' )
    os.mkdir(img_path + label_foldername)

    for n in range(len(image_sets)):
        path = image_sets[n]
        label_path = label_sets[n]

        # load the image
        inputdst = gdal.Open(img_path+path)
        if inputdst is None:
            print("can't open " + inputdst)
            return False

        # load the image
        labeldst = gdal.Open(img_path+label_path)
        if labeldst is None:
            print("can't open " + labeldst)
            return False

        # get the metadata of data
        w = inputdst.RasterXSize  # 原始栅格矩阵的列数
        h = inputdst.RasterYSize  # 原始栅格矩阵的行数
        c = inputdst.RasterCount
        padding_h = (h // stride + 1) * stride #+ (image_size - h % stride)
        padding_w = (w // stride + 1) * stride #+ (image_size - h % stride)
        if image_size < stride:
            print("image_size should larger than stride")
            break
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop = np.zeros((c,image_size,image_size),dtype=np.int16)
                crop_label = np.zeros((image_size, image_size),dtype=np.uint8)
                if j == padding_w // stride -1:
                    if i == padding_h // stride -1:
                        crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                    ysize=h % stride)
                        crop[:,0:h%stride,0:w%stride]=crop_tmp
                        label_tmp = labeldst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                        ysize=h % stride)
                        crop_label[0:h % stride, 0:w % stride] = label_tmp
                    else:
                        crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                    ysize=image_size)
                        crop[:, :, 0:w % stride] = crop_tmp
                        label_tmp = labeldst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=w % stride,
                                                        ysize=image_size)
                        crop_label[:, 0:w % stride] = label_tmp
                else:
                    if i == padding_h // stride -1:
                        crop_tmp = inputdst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=image_size,
                                                        ysize=h % stride)
                        crop[:, 0:h % stride, :] = crop_tmp
                        label_tmp = labeldst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=image_size,
                                                        ysize=h % stride)
                        crop_label[0:h % stride, :] = label_tmp
                    else:
                        crop = inputdst.ReadAsArray(xoff=j * stride,yoff=i * stride,xsize=image_size,ysize=image_size)
                        crop_label = labeldst.ReadAsArray(xoff=j * stride, yoff=i * stride, xsize=image_size,
                                                    ysize=image_size)

                crop_min=np.min(crop)
                if crop_min == 0:
                    #print('no invalid data!')
                    continue
                cb, ch, cw = crop.shape
                if ch != image_size or cw != image_size:
                    print('invalid size!')
                    continue
                if Augment == True:
                    crop, crop_label = data_augment2(crop, crop_label)
                tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
                rstDst = tifDriver.Create((img_path + 'src/'+headname+'%d.tiff' % g_count), img_w, img_h, cb, gdalconst.GDT_Int16);  # 创建目标文件
                for ii in range(cb):
                    rstDst.GetRasterBand(ii + 1).WriteArray(crop[ii])  # 写入数据
                rstlabelDst = tifDriver.Create((img_path + label_foldername +headname+'%d.tiff' % g_count), img_w, img_h, 1,
                                               gdalconst.GDT_Byte);  # 创建目标文件
                rstlabelDst.GetRasterBand(1).WriteArray(crop_label)  # 写入label数据
                g_count += 1
                print(g_count)

if __name__ == '__main__':
    #creat_dataset_random(20000,Augment=True)
    creat_dataset_regular(stridevalue=128,image_size=128,Augment=False,headname=head_name_seg)
    #sample_augment('C:/DB/01-ARMSMOTN/2019/train/')
