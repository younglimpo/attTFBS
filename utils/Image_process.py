#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2020/10/09 20:17
# @Author    :Lingbo Yang
# @Version   :v1.0 creat readimage
#    v1.1 cal confusion matrix
# @License: BSD 3 clause
# @Description: Image read, write
from osgeo import gdal,gdalconst
import numpy as np
from tqdm import tqdm
def readimage(imgpath):
    '''
    read image and return array
    :param imgpath: file path
    :return: np array
    '''
    inputdst = gdal.Open(imgpath)
    if inputdst is None:
        print("can't open " + imgpath)
        return False

    # get the metadata of data
    X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
    X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
    X_bands = inputdst.RasterCount # 原始栅格矩阵的band count

    src_img = inputdst.ReadAsArray(
        xoff=0, yoff=0, xsize=X_width, ysize=X_height)  # [X_height, X_width]
    src_img=np.array(src_img)
    return src_img

def saveImage(imgArr,imgPath,datatype,Xproj,XgeoInfo):
    '''
    Save image array to an image file
    :param imgArr: image arrary, c h w or h w
    :param imgPath: the output file path
    :param datatype: data type, like gdalconst.GDT_Int16
    :param Xproj: projection
    :param XgeoInfo: geo transformation
    :return:
    '''
    chw = imgArr.shape
    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
    if len(chw) == 2:
        rstDst = tifDriver.Create(imgPath, chw[1], chw[0], 1, datatype);  # 创建目标文件
        rstDst.SetGeoTransform(XgeoInfo)  # 写入仿射变换参数
        rstDst.SetProjection(Xproj)  # 写入投影
        rstDst.GetRasterBand(1).WriteArray(imgArr)  # 写入数据
    else:
        rstDst = tifDriver.Create(imgPath, chw[2], chw[1], chw[0], datatype);  # 创建目标文件
        rstDst.SetGeoTransform(XgeoInfo)  # 写入仿射变换参数
        rstDst.SetProjection(Xproj)  # 写入投影
        for i in range(chw[0]):
            rstDst.GetRasterBand(i + 1).WriteArray(imgArr[i])  # 写入数据

def saveImageArr2Txt(inImg,inLabel,txtpath, classlist=[0,1], retain_number = [10000, 10000]):

    X = readimage(inImg)  # 读取特征
    c, h, w, _, _ = getCHW(inImg)
    X = X.reshape(c, h * w)
    #X = X.transpose((1, 0))

    y = readimage(inLabel)
    y = y.reshape((h * w))
    y = y.reshape(1,(h * w))

    #把类别放在特征数据的最后一列，方便抽样
    xy = np.append(X,y, axis=0)
    xy = xy.transpose((1, 0))

    with open(txtpath, 'a') as f:
        #先抽取某一类别的所有像素，然后随机抽样其中的像素
        id = 0
        for i in classlist:
            xyi = xy[np.where(xy[:, -1] == i)] #最后一列是类别
            np.random.shuffle(xyi)
            np.savetxt(f, xyi[0:retain_number[id]])
            id = id+1

def tsne_drawTXT(inTxt,saveFig, classCount = 2, classList= ['Others', 'Rice'], s = 3, alf=0.3):
    #import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets
    import datetime

    starttime = datetime.datetime.now()
    # classCount = 4 #类别的数量
    # classList = ['Corn','Cotton','Rice','Soybeans']

    '''t-SNE'''
    XY = np.loadtxt(inTxt)
    x = XY[:,0:-1]
    y = XY[:,-1]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    #tsne = manifold.TSNE(n_components=2,  random_state=501)
    x_tsne = tsne.fit_transform(x)
    # long running
    endtime = datetime.datetime.now()
    print('SNEtime cost:')
    print(endtime - starttime)

    '''嵌入空间可视化'''
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    colorlist = ['g', 'b']

    for i in range(classCount):
        class_tmp_index = np.where(y == i)
        x_tmp = (X_norm[class_tmp_index, :])  # 会多出一个1维来
        plt.scatter(x_tmp[0, :, 0], x_tmp[0, :, 1], c=colorlist[i], s=s, label=classList[i], alpha=alf)

    bwith = 4
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=19)
    endtime2 = datetime.datetime.now()
    print('plt time cost:')
    print(endtime2 - starttime)
    plt.tight_layout()
    plt.savefig(saveFig, dpi=300, figsize=(300, 300))
    print('done ' + saveFig)
    # plt.show()


def adjustimage(imgpath,outpath,proportion):
    '''
    adjust the image based on an array and save it
    :param imgpath: file path
    :return: np array
    '''
    inputdst = gdal.Open(imgpath)
    if inputdst is None:
        print("can't open " + imgpath)
        return False

    # get the metadata of data
    X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
    X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
    X_bands = inputdst.RasterCount # 原始栅格矩阵的band count
    adfGeoTransform = inputdst.GetGeoTransform()
    X_prj = inputdst.GetProjection()

    src_img = inputdst.ReadAsArray(
        xoff=0, yoff=0, xsize=X_width, ysize=X_height)  # [X_height, X_width]
    src_img=np.array(src_img)

    for i in range(X_bands):
        src_img[i,...] = src_img[i,...] * proportion[i]

    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
    rstDst = tifDriver.Create(outpath, X_width, X_height, X_bands, gdalconst.GDT_Int16);  # 创建目标文件
    rstDst.SetGeoTransform(adfGeoTransform)  # 写入仿射变换参数
    rstDst.SetProjection(X_prj)  # 写入投影
    for i in range(X_bands):
        rstDst.GetRasterBand(i + 1).WriteArray(src_img[i])  # 写入数据

    '''inimg = r'D:\03-SiameseNet\04-2019SC\07-2020SC\2020-SC-VV-VH.dat'
    outimg = r'D:\03-SiameseNet\04-2019SC\07-2020SC\2020-SC-VV-VH-adjust.dat'
    propotion_arr = [1.412555525,
                     0.909586589,
                     1.09935482,
                     1.057446692,
                     0.85031338,
                     0.934647149,
                     0.881468607,
                     0.99525722,
                     1.147829153,
                     1.346665943,
                     0.947235873,
                     0.993550618,
                     1.061088853,
                     0.945109618,
                     0.968989057,
                     0.893595764,
                     0.98250366,
                     1.14122171
                     ]
    adjustimage(inimg, outimg, propotion_arr)
    print('done!')'''


def getCHW(imgpath):
    '''
    GET c h w of an image
    :param imgpath:
    :return: [c h w prj geoTrans]
    '''
    inputdst = gdal.Open(imgpath)
    if inputdst is None:
        print("can't open " + imgpath)
        return False

    # get the metadata of data
    X_width = inputdst.RasterXSize  # 原始栅格矩阵的列数
    X_height = inputdst.RasterYSize  # 原始栅格矩阵的行数
    X_bands = inputdst.RasterCount  # 原始栅格矩阵的band count
    X_prj = inputdst.GetProjection()  # 获取投影类型
    adfGeoTransform = inputdst.GetGeoTransform()  # 获取原始数据的转换7参数

    return [X_bands,X_height,X_width,X_prj,adfGeoTransform]

def calProducerAccuracy(y_true,y_pred):
    '''
    calculate producer's accuracy or recall
    :param y_true:
    :param y_pred:
    :return: recall or producer's accuracy
    '''
    TP = np.sum(y_true * y_pred)
    recall = TP / (np.sum(y_true)+0.0000001)  # equivalent to the above two lines of code
    return recall

def calUserAccuracy(y_true,y_pred):
    '''
    calculate user's accuracy or precision
    :param y_true:
    :param y_pred:
    :return: precision or user's accuracy
    '''
    TP = np.sum(y_true * y_pred)
    precision = TP / (np.sum(y_pred) + 0.0000001)  # equivalent to the above two lines of code
    return precision

def calConfusionMatrix(PredictImg,BaseImg):
    PredArray=readimage(PredictImg)
    BaseArray=readimage(BaseImg)
    prodacc=calProducerAccuracy(BaseArray,PredArray)
    useracc = calUserAccuracy(BaseArray, PredArray)
    kappa=calkappa(BaseArray, PredArray)
    fscore=calfmeasure(BaseArray, PredArray)
    return [fscore,kappa,prodacc,useracc]

def calfmeasure(y_true,y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    TP = np.sum(y_true * y_pred)
    precision = TP / (np.sum(y_pred)+ 0.0000001)
    recall = TP / (np.sum(y_true)+ 0.0000001)
    F1score = 2 * precision * recall / (precision + recall+0.0000001)
    return F1score

def calkappa(y_true, y_pred):
    # Calculates the kappa coefficient
    TP = np.float(np.sum(y_true *y_pred))
    FP = np.float(np.sum((1 - y_true) * y_pred))
    FN = np.float(np.sum(y_true * (1 - y_pred)))
    TN = np.float(np.sum((1 - y_true) * (1 - y_pred)))
    totalnum=TP+FP+FN+TN
    p0 = (TP + TN)/ (totalnum+0.0000001)
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN))/ (totalnum * totalnum+0.0000001)
    kappa_coef = (p0 - pe)/(1 - pe + 0.0000001)
    return kappa_coef

def calConfusionMatrix_2(PredictImg,BaseImg,classlist=[0,1,2,3]):
    '''
    计算混淆矩阵，可以计算类别数量不相等的图像
    :param PredictImg:
    :param BaseImg:
    :param classlist:
    :return: 混淆矩阵 np array，行方向为预测值，列方向为预测值
    '''
    PredArray = readimage(PredictImg)
    BaseArray = readimage(BaseImg)
    class_nb = len(classlist)
    ConfusionMatrix = np.zeros(shape=(class_nb,class_nb))
    id_i = 0
    for i in classlist:
        base_i = (BaseArray == i)
        id_j = 0
        for j in classlist:
            pred_j = (PredArray == j)
            base_ij = base_i * pred_j # 参考值为i且预测值为j的像元
            count_ij = np.sum(base_ij)
            ConfusionMatrix[id_i, id_j] = count_ij
            id_j = id_j + 1
        id_i = id_i + 1
    return ConfusionMatrix


def summaryBinary2Multi(inputTifList,output,weight=None,thresholdList = None):
    '''
    transferm multi binary classification to one multiclass classification result
    :param inputTifList: an list composed of binary classification resluts' filepath
    :param output: output classification result
    :param weight: the weight of each binary classification result, if None, will equal to 1
    :param threshold: the threshold of binary classification results, is scaled by 100, then equal 50
    :return:
    '''
    chw = getCHW(inputTifList[0])
    img_total = np.zeros((len(inputTifList)+1,chw[1],chw[2]), dtype=np.float16) #增加1列代表0，其他类型
    for i, imgpath in enumerate(tqdm(inputTifList)):
        img = readimage(imgpath)
        threshold=50
        if thresholdList is not None:
            threshold = thresholdList[i]
        mask = (img >= threshold)
        weight_i = 1.
        if weight is not None:
            weight_i = weight[i]
        img = img * mask * weight_i
        img_total[i+1,...] = img #使其在0时表示其他，第一类时则为1

    class_img = np.argmax(img_total,axis=0)


    inputdst = gdal.Open(inputTifList[0])

    # get the metadata of data
    X_prj = inputdst.GetProjection()  # 获取投影类型
    adfGeoTransform = inputdst.GetGeoTransform()  # 获取原始数据的转换7参数

    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式

    rstlabelDst = tifDriver.Create(output, chw[2], chw[1], 1,
                                       gdalconst.GDT_Byte);  # 创建目标文件

    rstlabelDst.SetGeoTransform(adfGeoTransform)  # 写入仿射变换参数
    rstlabelDst.SetProjection(X_prj)  # 写入投影
    rstlabelDst.GetRasterBand(1).WriteArray(class_img)  # 写入label数据

def summaryBinary2Multi_2(inputTifList,output,maskfile, weight=None):
    '''
    transferm multi binary classification to one multiclass classification result
    :param inputTifList: an list composed of binary classification resluts' filepath
    :param output: output classification result
    :param weight: the weight of each binary classification result, if None, will equal to 1
    :param threshold: the threshold of binary classification results, is scaled by 100, then equal 50
    :return:
    '''
    chw = getCHW(inputTifList[0])
    img_total = np.zeros((len(inputTifList)+1,chw[1],chw[2]), dtype=np.float16) #增加1列代表0，其他类型
    for i, imgpath in enumerate(tqdm(inputTifList)):
        img = readimage(imgpath)
        mask = readimage(maskfile)
        weight_i = 1.
        if weight is not None:
            weight_i = weight[i]
        img = img * mask * weight_i
        img_total[i+1,...] = img #使其在0时表示其他，第一类时则为1

    class_img = np.argmax(img_total,axis=0)


    inputdst = gdal.Open(inputTifList[0])

    # get the metadata of data
    X_prj = inputdst.GetProjection()  # 获取投影类型
    adfGeoTransform = inputdst.GetGeoTransform()  # 获取原始数据的转换7参数

    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式

    rstlabelDst = tifDriver.Create(output, chw[2], chw[1], 1,
                                       gdalconst.GDT_Byte);  # 创建目标文件

    rstlabelDst.SetGeoTransform(adfGeoTransform)  # 写入仿射变换参数
    rstlabelDst.SetProjection(X_prj)  # 写入投影
    rstlabelDst.GetRasterBand(1).WriteArray(class_img)  # 写入label数据

def ConvertEachBand2File(inputfile,outfileList,DataType):
    '''
    Export every band of the input img to an indenpendant file
    :param inputfile:
    :param outfileList:
    :param DataType: gdalconst.
    :return:
    '''

    rawImg = readimage(inputfile)

    CHW = getCHW(inputfile)
    if len(outfileList) != CHW[0]:
        print('the len of outputfile list'+str(len(outfileList))
              +' should be equal to the band account of input image'+ str(CHW[0])+'!')
        return
    for i in tqdm(range(CHW[0])):
        saveImage(rawImg[i,...],outfileList[i],DataType,CHW[3],CHW[4])

def ConvertEachValue2File(inputfile,outfileList,valueCount):
    '''
    Export each value of an one band input img to an indenpendant file, except 0
    :param inputfile:
    :param outfileList:
    :param valueCount: 4类就是4
    :return:
    '''


    rawImg = readimage(inputfile)

    CHW = getCHW(inputfile)
    if CHW[0] != 1:
        print('the dims of input image should be equal to 2!')
        return
    for i in tqdm(range(valueCount)):
        rawImg_i = (rawImg == (i+1))
        saveImage(rawImg_i,outfileList[i],gdalconst.GDT_Byte,CHW[3],CHW[4])


def TSNEMapGenerater(inputfile,inputLabel,classCount,classList):#
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets
    import datetime

    starttime = datetime.datetime.now()
    #classCount = 4 #类别的数量
    #classList = ['Corn','Cotton','Rice','Soybeans']

    X = readimage(inputfile) #读取特征
    c,h,w,_,_ = getCHW(inputfile)
    X = X.reshape(c,h*w)
    X = X.transpose((1,0))

    y = readimage(inputLabel)
    y = y.reshape((h*w))




    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca',  random_state=501)
    X_tsne = tsne.fit_transform(X)
    np.savetxt(r"d:\X-timefeature-vv_vh_TSNE_X.txt", X_tsne)
    #X_tsne = np.loadtxt(r"d:\X-timefeature-vv_vh_TSNE_X.txt")
    np.savetxt(r"d:\X-timefeature-vv_vh_TSNE_y.txt", y)
    #print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    # long running
    endtime = datetime.datetime.now()
    print('SNEtime cost:')
    print(endtime - starttime)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    colorlist=['r','g','b','y']

    for i in range(classCount):
        class_tmp_index = np.where(y == (i+1))
        x_tmp = (X_norm[class_tmp_index,:]) #会多出一个1维来
        #plt.scatter(x_tmp[0,:, 0], x_tmp[0,:, 1], c=plt.cm.Set1(i),s=1, label=classList[i])
        plt.scatter(x_tmp[0,:, 0], x_tmp[0,:, 1], c=colorlist[i],s=1, label=classList[i],alpha=0.5)



    '''for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1],c=plt.cm.Set1(y[i]),label=str(y[i]))
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})'''
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=14)
    endtime2 = datetime.datetime.now()
    print('plt time cost:')
    print(endtime2 - starttime)
    plt.tight_layout()
    plt.savefig('d:/timefeature-vv-vh-tsne.png', dpi=300, figsize=(300, 300))
    plt.show()

def TSNEMapGenerater_2Class(inputfile,inputLabel,classCount,classList,saveXtxt=r'd:\x.txt',saveYtxt=r'd:\y.txt',saveFig=r'd:\t-sne.txt',isLoad=False,s = 3, alf=0.3):#
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets
    import datetime

    starttime = datetime.datetime.now()
    #classCount = 4 #类别的数量
    #classList = ['Corn','Cotton','Rice','Soybeans']


    '''t-SNE'''
    if isLoad:
        X_tsne = np.loadtxt(saveXtxt)
        y = np.loadtxt(saveYtxt)
    else:
        X = readimage(inputfile)  # 读取特征
        c, h, w, _, _ = getCHW(inputfile)
        X = X.reshape(c, h * w)
        X = X.transpose((1, 0))

        y = readimage(inputLabel)
        y = y.reshape((h * w))
        tsne = manifold.TSNE(n_components=2, init='pca',  random_state=501)
        X_tsne = tsne.fit_transform(X)
        np.savetxt(saveXtxt, X_tsne)
        np.savetxt(saveYtxt, y)

    # long running
    endtime = datetime.datetime.now()
    print('SNEtime cost:')
    print(endtime - starttime)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    colorlist=['g','b']

    for i in range(classCount):
        class_tmp_index = np.where(y == i)
        x_tmp = (X_norm[class_tmp_index,:]) #会多出一个1维来
        plt.scatter(x_tmp[0,:, 0], x_tmp[0,:, 1], c=colorlist[i],s=s, label=classList[i],alpha=alf)

    bwith=4
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=19)
    endtime2 = datetime.datetime.now()
    print('plt time cost:')
    print(endtime2 - starttime)
    plt.tight_layout()
    plt.savefig(saveFig, dpi=300, figsize=(300, 300))
    print('done '+saveFig)
    #plt.show()

def TSNE_MAIN():
    inputfile = r'E:\11-ArMsMoTn\11-TSNE\2019AR-4class-VVVH-pred-timeFeatures.dat'
    inputLabel = r'E:\11-ArMsMoTn\11-TSNE\2019AR-4class-label.dat'
    classCount = 4
    classList = ['Corn', 'Cotton', 'Rice', 'Soybeans']
    TSNEMapGenerater(inputfile, inputLabel, classCount, classList)

def TSNE_LSTM_MAIN():
    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\LSTM\2017-ROI-LSTM-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\2017-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\LSTM\01-2017-LSTM-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\LSTM\01-2017-LSTM-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\LSTM\01-2017-LSTM-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others',  'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig,isLoad=True,s=8,alf=0.3)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\LSTM\2018-ROI-LSTM-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\2018-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\LSTM\01-2018-LSTM-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\LSTM\01-2018-LSTM-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\LSTM\01-2018-LSTM-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig,isLoad=True,s=8,alf=0.3)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\LSTM\2019-ROI-LSTM-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\2019-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\LSTM\01-2019-LSTM-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\LSTM\01-2019-LSTM-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\LSTM\01-2019-LSTM-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig,isLoad=True,s=8,alf=0.3)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\LSTM\2020-ROI-LSTM-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\2020-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\LSTM\01-2020-LSTM-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\LSTM\01-2020-LSTM-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\LSTM\01-2020-LSTM-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig,isLoad=True,s=8,alf=0.3)

def TSNE_UNET_MAIN():
    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\UNET\2017-ROI-UNET-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\2017-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\UNET\01-2017-UNET-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\UNET\01-2017-UNET-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\UNET\01-2017-UNET-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others',  'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\UNET\2018-ROI-UNET-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\2018-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\UNET\01-2018-UNET-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\UNET\01-2018-UNET-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\UNET\01-2018-UNET-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\UNET\2019-ROI-UNET-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\2019-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\UNET\01-2019-UNET-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\UNET\01-2019-UNET-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\UNET\01-2019-UNET-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\UNET\2020-ROI-UNET-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\2020-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\UNET\01-2020-UNET-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\UNET\01-2020-UNET-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\UNET\01-2020-UNET-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

def TSNE_ConvLSTM_MAIN():
    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\ConvLSTM\2017-ROI-ConvLSTM-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\2017-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\ConvLSTM\01-2017-ConvLSTM-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\ConvLSTM\01-2017-ConvLSTM-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\ConvLSTM\01-2017-ConvLSTM-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others',  'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\ConvLSTM\2018-ROI-ConvLSTM-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\2018-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\ConvLSTM\01-2018-ConvLSTM-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\ConvLSTM\01-2018-ConvLSTM-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\ConvLSTM\01-2018-ConvLSTM-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\ConvLSTM\2019-ROI-ConvLSTM-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\2019-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\ConvLSTM\01-2019-ConvLSTM-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\ConvLSTM\01-2019-ConvLSTM-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\ConvLSTM\01-2019-ConvLSTM-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\ConvLSTM\2020-ROI-ConvLSTM-afterFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\2020-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\ConvLSTM\01-2020-ConvLSTM-afterFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\ConvLSTM\01-2020-ConvLSTM-afterFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\ConvLSTM\01-2020-ConvLSTM-afterFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

def TSNE_TFBS_MAIN():
    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\TFBS\2017-ROI-TFBS-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\2017-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\TFBS\01-2017-TFBS-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\TFBS\01-2017-TFBS-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2017\TFBS\01-2017-TFBS-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others',  'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList,outX,outY,outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\TFBS\2018-ROI-TFBS-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\2018-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\TFBS\01-2018-TFBS-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\TFBS\01-2018-TFBS-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2018\TFBS\01-2018-TFBS-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\TFBS\2019-ROI-TFBS-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\2019-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\TFBS\01-2019-TFBS-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\TFBS\01-2019-TFBS-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2019\TFBS\01-2019-TFBS-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

    inputfile = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\TFBS\2020-ROI-TFBS-beforeFine-feature.dat'
    inputLabel = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\2020-ROI-label.dat'
    outX = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\TFBS\01-2020-TFBS-beforeFine-features-tSNE-X.txt'
    outY = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\TFBS\01-2020-TFBS-beforeFine-features-tSNE-Y.txt'
    outFig = r'E:\11-ArMsMoTn\04-Dataset\15-SC\06-t-SNE\2020\TFBS\01-2020-TFBS-beforeFine-features-tSNE.png'
    classCount = 2
    classList = ['Others', 'Rice', ]
    TSNEMapGenerater_2Class(inputfile, inputLabel, classCount, classList, outX, outY, outFig)

def calaccuracy_main():
    out_img1 = r'F:\01-NorthEast\04-DB-P\03-classification\2017\liaohe\辽河attTFBS分类结果RECLASS.tif'

    #inputTif = r'E:\11-ArMsMoTn\04-Dataset\13-CA\2019\02-Classification\01-OneSample\02-LSTM\2019-CA-LSTM-OneSample_url1109.tiff_20201016-111351-pred.tif'
    baseImg = r'F:\01-NorthEast\04-DB-P\04-NorthEastRiceRef\rice-clip\2017\liaohe\辽河参考resize.tif'
    accuracy = calConfusionMatrix_2(out_img1, baseImg)
    print(accuracy)



def calmultiaccuracy_main():
    '''in1 = r'E:\11-ArMsMoTn\04-Dataset\03-2018Dataset\01-multiClass-result\01-UNET\AR2019AugTrain-AR18Test-UNET-Corn_soft.tif'
        in2 = r'E:\11-ArMsMoTn\04-Dataset\03-2018Dataset\01-multiClass-result\01-UNET\AR2019AugTrain-AR18Test-UNET-Cotton_soft.tif'
        in3 = r'E:\11-ArMsMoTn\04-Dataset\03-2018Dataset\01-multiClass-result\01-UNET\AR2019AugTrain-AR18Test-UNET-Rice_soft.tif'
        in4 = r'E:\11-ArMsMoTn\04-Dataset\03-2018Dataset\01-multiClass-result\01-UNET\AR2019AugTrain-AR18Test-UNET-Soybeans_soft.tif'

        output = r'E:\11-ArMsMoTn\04-Dataset\03-2018Dataset\01-multiClass-result\01-UNET\AR2019AugTrain-AR18Test-UNET-Summary-class.tif  '
        inlist = [in1,in2,in3,in4]

        weight = [0.768078327178955,0.774898648262023,0.873196482658386,0.879209816455841] #unet precision

        #weight = [7.748637199401855469e-01,6.953886747360229492e-01,8.923116922378540039e-01,8.635920286178588867e-01] #tfbs precision

        #weight = [4.941075146198273260e-01,3.070266544818878174e-01,5.671368837356567383e-01,7.880997061729431152e-01] #LSTM precision
        #weight = [7.275457382202148438e-01,5.522915720939636230e-01,8.277832269668579102e-01,8.717470169067382812e-01] #LSTM precision
        #thresholdlist=[50,20,50,20]
        summaryBinary2Multi(inlist,output,weight=weight) #thresholdList = thresholdlist,
        print('Done!')'''

def imageAug(inImg,inLabel,outImg,valueArr):
    inImgArr = readimage(inImg)
    inLabelArr = readimage(inLabel)
    c,h,w,prj,geo =getCHW(inImg)
    for i in range(c):
        addImg=inLabelArr*valueArr[i]
        inImgArr[i,...]=inImgArr[i,...]+addImg
    saveImage(inImgArr,outImg,gdalconst.GDT_Int16,prj,geo)


def sampleMain():
    #用于抽样一定的像素
    inImg = 'F:/01-NorthEast/01-TSNE/03-SC/2020/FinedFeature/2020-ROI-VVVH-FinedFeatures.dat'
    inLable = 'F:/01-NorthEast/01-TSNE/03-SC/2020/2020-ROI-label.dat'
    txtpath = 'F:/01-NorthEast/01-TSNE/03-SC/2020/FinedFeature/2020-ROI-VVVH-FinedFeatures.txt'
    saveImageArr2Txt(inImg, inLable, txtpath)

def tsne_main():
    #用于tsne绘图
    txtpath = 'F:/01-NorthEast/01-TSNE/03-SC/2020/FinedFeature/2020-ROI-VVVH-FinedFeatures.txt'
    saveFig = 'F:/01-NorthEast/01-TSNE/03-SC/2020/FinedFeature/2020-ROI-VVVH-FinedFeatures.jpg'
    tsne_drawTXT(txtpath, saveFig, classCount=2, classList=['Others', 'Rice'], s=5, alf=0.3)

if __name__=='__main__':
    calaccuracy_main()
    #sampleMain()
    #tsne_main()










