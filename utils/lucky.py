#!usr/bin/env python
# -*- coding: utf-8 -*-
'''
v1.0,2017/11/17/17:23 by ylb
v1.1,2017/11/24/21:03 by ylb
v1.2,2017/11/25/09:56 by ylb
v1.3 2019/9/27 修改了格式转换中的范围
v1.4 20200925 添加读取excel文件指定单元格数据的功能
v1.5 20200909   添加读取excel某一列最大值并返回索引的功能
v1.6 20201030 添加读取jpg文件保存到txt功能
'''
import os
import datetime
import numpy as np
from glob import glob
from math import ceil
import time
from osgeo import gdal
def calmath(str,**x):
    '''
    输入计算公式字符串，以及公式中各变量的定义值，返回计算结果
    :param str:计算公式字符串，例：'a+b*c'
    :param x:与公式对应的变量定义，例：a=2,b=3,c=4
    :return:计算结果
    '''
    #print(str)
    varname=locals()#动态变量
    for m in x: #动态变量赋值
        varname['%s' %m] = x[m]
    rst=eval(str)
    #print rst
    return rst

def readXLXSdatatoArr(excel_path,readcolds = [0,1,2,3,4,5,6,7,8]):
    import pandas as pd
    df = pd.read_excel(excel_path, usecols=readcolds,names=None)  # 读取项目名称和行业领域两列，并不要列名
    df_li = df.values.tolist()
    return df_li

def Julian2Date(year,day):
    '''
    将儒略历转为公历年月日
    :param year:年
    :param day:第几天
    :return:公历日期
    '''
    fir_day = datetime.datetime(year,1,1)
    zone = datetime.timedelta(days=day-1)
    return datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")

def Date2Julian(YMD):
    '''
    将年月日转换为儒略日
    :param YMD: 如20170514
    :return: 如：134
    '''
    y = int(YMD[0:4])  # 获取年份
    m = int(YMD[4:6])  # 获取月份
    d = int(YMD[6:8])  # 获取“日”
    dt = datetime.datetime(y, m, d)
    return dt.strftime("%j")


def getFileUnderPath(path,extension,Issubfolder=False,fullName=True,returnExtension=True):
    '''
    获取目录和子目录下所有制定后缀的文件
    :param path:目录,如“C:\\Temp\\”
    :param extension:后缀
    :param Issubfolder:是否搜索子目录
    :return:返回list
    '''
    list_file = []
    # 获取指定目录下一级目录的所有指定后缀的文件名
    if Issubfolder!=True:
        f_list = os.listdir(path)
        # print f_list
        for i in f_list:
            # os.path.splitext():分离文件名与扩展名
            if os.path.splitext(i)[1] == extension:
                if not returnExtension:
                    i = os.path.splitext(i)[0]
                filePath_tmp = os.path.join(path, i)  #路径拼接
                if fullName:
                    list_file.append(filePath_tmp)
                else:
                    list_file.append(i)
                #list_file.append(path+i)
    #获取目录下所有子目录下的所有的文件
    if Issubfolder==True:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                fullfilename=os.path.join(dirpath, filename)
                if os.path.splitext(fullfilename)[1] == extension:
                    list_file.append(fullfilename)
    return list_file


def renameFileExtension(filename,rawEXT,newEXT,isRenameFile=False):
    '''
    用来修改文件名的后缀名
    :param filename: 文件名如‘D:/ABC/’
    :param rawEXT: 原扩展名，如“.dat”
    :param newEXT: 修改后扩展名，如“.tif”
    :param isRenameFile:,是否需要修改源文件名称，默认为否
    :return: 修改后的文件名
    '''
    portion = os.path.splitext(filename)
    # 如果后缀是.dat
    if portion[1] == rawEXT:
        # 重新组合文件名和后缀名
        newname = portion[0] + newEXT
        if isRenameFile:
            os.rename(filename, newname)
        return newname

def rasterFormatConvert(inputfile,outputfile):
    import gdal
    Srcdst = gdal.Open(inputfile)
    if Srcdst == None:
        print("can't open " + inputfile)
        return False
    im_width = Srcdst.RasterXSize  # 原始栅格矩阵的列数
    im_height = Srcdst.RasterYSize  # 原始栅格矩阵的行数
    im_bands = Srcdst.RasterCount  # 原始波段数
    im_eDT = Srcdst.GetRasterBand(1).DataType;  # 数据的类型
    im_prj = Srcdst.GetProjection()  # 获取投影
    adfGeoTransform = Srcdst.GetGeoTransform()  # 获取原始数据的转换7参数
    im_data = Srcdst.ReadAsArray(0, 0, im_width, im_height)  # 获取数据

    tifDriver = gdal.GetDriverByName("GTiff");  # 目标格式
    rstDst = tifDriver.Create(outputfile, im_width, im_height, im_bands, im_eDT);  # 创建目标文件
    rstDst.SetGeoTransform(adfGeoTransform)  # 写入仿射变换参数
    rstDst.SetProjection(im_prj)  # 写入投影

    for num in range(0,im_bands):
        if im_bands == 1:
            data = im_data
        else:
            data = im_data[num, :, :]

        nodata = Srcdst.GetRasterBand(num+1).GetNoDataValue();
        if nodata:
            rstDst.GetRasterBand(num+1).SetNoDataValue(nodata);  # 设置无效值
        rstDst.GetRasterBand(num+1).WriteArray(data)  # 写入数据


def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y


def RasterMosaic(file_list, outpath):
    Open = gdal.Open
    min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)
        min_x, min_y = min(min_x, minx), min(min_y, miny)
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)

    in_ds = Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)  # 列数
    rows = ceil((max_y - min_y) / (-height))  # 行数

    outfile = outpath + file_list[0][:4] + '.tif'  # 结果文件名，可自行修改
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outfile, columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x  # 更正左上角坐标
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    for in_fn in file_list:
        in_ds = Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)

        data = in_ds.GetRasterBand(1).ReadAsArray()
        out_band.WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
    del in_ds, out_band, out_ds
    return outfile


def compress2(path, target_path, method="LZW"):  #
    """使用gdal进行文件压缩，
          LZW方法属于无损压缩，
          效果非常给力，4G大小的数据压缩后只有三十多M"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, callback=progress,options=["TILED=YES", "COMPRESS={0}".format(method)])
    del dataset

def mosaicMain():
    path = r'J:\backup' #该文件夹下存放了待拼接的栅格
    os.chdir(path)
    raster_list = sorted(glob('*.tif')) #读取文件夹下所有tif数据
    result = RasterMosaic(raster_list,outpath = r'J:\backup\Global' ) #拼接栅格
    compress2(result,target_path = r'J:\backup\Global.tif') #压缩栅格

def getXLXSMaxRowIndex(excel_path, readbyname=True, col_nb=3, strow=1, endrow=31, sheetname='', sheetindex=0):
        '''
        读取excel的指定列的最大值的行号
        :param excel_path: 文件路径
        :param readbyname: True为名称读取sheet，False为索引读取sheet
        :param col_nb: 获取最大值的列索引
        :param strow: 开始读取的行索引
        :param endrow: 结束读取的行索引
        :param sheetname: 表名
        :param sheetindex: sheet索引
        :return: 最大值所在行列号
        '''
        import xlrd
        from xlrd import xldate_as_tuple
        from datetime import datetime
        excel_file = xlrd.open_workbook(excel_path)
        if readbyname:
            table = excel_file.sheet_by_name(sheetname)  # 通过名字打开
        else:
            table = excel_file.sheet_by_index(sheetindex)  # 通过索引打开
        col_data = table.col_values(col_nb, strow, endrow)

        maxindex=col_data.index(max(col_data))+strow

        return maxindex

def readXLXSdata(excel_path,readbyname = True,cell_row=1,cel_col=1,sheetname='',sheetindex=0):
    '''
    读取excel的指定单元格
    :param excel_path: 文件路径
    :param readbyname: True为名称读取sheet，False为索引读取sheet
    :param cell_row: 行
    :param cel_col: 列
    :param sheetname: sheet名
    :param sheetindex: sheet索引
    :return: 单元格值
    '''
    import xlrd
    from xlrd import xldate_as_tuple
    from datetime import datetime
    excel_file = xlrd.open_workbook(excel_path)
    if readbyname:
        table = excel_file.sheet_by_name(sheetname)   # 通过名字打开
    else:
        table = excel_file.sheet_by_index(sheetindex) # 通过索引打开
    ctype = table.cell(cell_row, cel_col).ctype  # 获取单元格返回的数据类型
    cell_value = table.cell(cell_row, cel_col).value  # 获取单元格内容
    if ctype == 2:  # 是否是数字类型 and cell_value % 1 == 0.00
        cell_value = float(cell_value)
    elif ctype == 3:  # 是否是日期
        date = datetime(*xldate_as_tuple(cell_value, 0))
        cell_value = date.strftime('%Y/%m/%d %H:%M:%S')
    elif ctype == 4:  # 是否是布尔类型
        cell_value = True if cell_value == 1 else False
    return cell_value

def saveList2TXT(listvalue,txtpath):
    '''
    保存list数组数据到txt中
    :param listvalue: list数据如，[1,2,3,4]
    :param txtpath: 保存路径，如‘D:/123.txt’
    :return:
    '''
    file = open(txtpath, 'w');

    file.write(str(listvalue));

    file.close();

def saveList2TXT_line(listvalue,txtpath):
    '''
    保存list数组数据到txt中
    :param listvalue: list数据如，[1,2,3,4]
    :param txtpath: 保存路径，如‘D:/123.txt’
    :return:
    '''
    file = open(txtpath, 'w');
    for i in listvalue:
        file.write(str(i)+'\r');
    file.close();

def saveList2TXT_append(listvalue,txtpath):
    '''
    保存list数组数据到txt中
    :param listvalue: list数据如，[1,2,3,4]
    :param txtpath: 保存路径，如‘D:/123.txt’
    :return:
    '''
    file = open(txtpath, 'a');
    for i in listvalue:
        file.write(str(i)+'  ');
    file.write('\r')
    file.close();

def file_move(inpath,outpath,percent=0.3,islable=True):
    """
    随机移动指定比例的文件到指定文件夹
    :param inpath:
    :param outpath:
    :param percent:
    :param islable: 是否移动label文件夹的文件
    :return:
    """
    import glob,shutil,random
    from tqdm import tqdm
    file_list = glob.glob(inpath + "/*.tiff")
    random.shuffle(file_list)
    lenth_list = len(file_list)
    move_count = lenth_list*percent
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    if islable:
        outpath_label = outpath.replace('src','label')
        if not os.path.exists(outpath_label):
            os.mkdir(outpath_label)
    for i in tqdm(range(len(file_list))):
        if i < move_count:
            shutil.move(file_list[i], outpath)
            if islable:
                label = file_list[i].replace('src','label')
                shutil.move(label, outpath_label)



def saveArray2TXT(npArray,txtpath):
    '''
    保存np array数组数据到txt中
    :param npArray: array数据如，[1,2,3,4]
    :param txtpath: 保存路径，如‘D:/123.txt’
    :return:
    '''
    np.savetxt(txtpath,npArray)



def csv_to_xlsx_pd(csvpath,xlsxpath):
    import pandas as pd
    csv = pd.read_csv(csvpath, encoding='utf-8')
    csv.to_excel(xlsxpath)

def makeTxtFile(folderpath,outputtxt,extension='.tiff'):
    '''
    将所有目录下的制定后缀文件写入txt中
    :param folderpath:
    :param outputtxt:
    :param extension:
    :return:
    '''
    filelist=getFileUnderPath(folderpath,extension=extension,fullName=False,returnExtension=False)
    saveList2TXT_line(filelist,outputtxt)
    print('done')

def renameFileList(filelist,renameFilelist):
    '''
    用来批量重命名文件
    :param filelist: 文件名列表
    :param renameFilelist: 重命名的文件名列表
    :return:
    '''
    i = 0
    for m in filelist:
        os.rename(m,renameFilelist[i])
        i=i+1
    print('done!')

def progress(percent, msg, tag):
    """进度回调函数"""
    print(percent, msg, tag)


def compress(path, target_path):
    """使用gdal进行文件压缩"""
    import gdal
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, callback=progress, options=["TILED=YES", "COMPRESS=PACKBITS"])
    # strict=1表示和原来的影像严格一致，0表示可以有所调整
    # callback为进度回调函数
    # PACKBITS快速无损压缩，基于流
    # LZW针对像素点，黑白图像效果好
    del dataset

def drawConfusionMatrix(confusionMatrix, output, xLabel = 'Prediction of TFBS', IsAnnot = True, IsCbar = True, classlist=['Others','Corn','Cotton','Rice','Soybeans']):
    '''
    绘制混淆举证热力图
    :param confusionMatrix:
    :param output:
    :param Mapname:
    :param IsAnnot:
    :param IsCbar:
    :param classlist:
    :return:
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    xtick = classlist
    ytick = classlist

    fig, ax = plt.subplots(figsize=(7, 6))
    #ax.set_title('Correlation between features')

    plt.tick_params(labelsize=15)
    sns.set(font_scale=1.2)
    h = sns.heatmap(confusionMatrix, fmt='g', annot=IsAnnot, cbar=False, xticklabels=xtick,
                yticklabels=ytick)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒,cmap= 'RdBu',
    cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小


    ax.set_ylabel('CDL', fontsize=15)
    ax.set_xlabel(xLabel, fontsize=15)

    fig.tight_layout()

    plt.savefig(output, dpi=300)
    plt.show()

def ConfusionMap():
    CM_TFBS = [[97.13/100,0.14/100,0.18/100,0.12/100,2.44/100],[17.55/100,75.43/100,0.29/100,0.99/100,5.74/100],[2.42/100,0.32/100,66.91/100,0.19/100,30.16/100],
          [ 7.95/100,0.34/100,0.12/100,  87.10/100 ,4.49/100],[12.65/100,3.10/100 ,1.73/100  ,1.48/100  ,81.03/100]] #TFBS
    CM_UNET = [[98.80/100,0.13/100,0.11/100,0.15/100,0.81/100],[22.12/100,74.94/100,0.08/100,0.91/100,1.95/100],[11.64/100,0.28/100,62.00/100,0.27/100,25.81/100],
          [ 11.73/100,0.40/100,0.05/100,  86.04/100 ,1.78/100],[26.90/100,1.80/100 ,1.49/100  ,1.85/100  ,67.95/100]] #UNET

    CM_LSTM = [[95.84 / 100, 0.21 / 100, 0.09 / 100, 0.19 / 100, 3.67 / 100],
               [14.99 / 100, 67.39 / 100, 0.30 / 100, 1.37 / 100, 15.95 / 100],
               [1.32 / 100, 0.61 / 100, 30.15 / 100, 0.89 / 100, 67.03 / 100],
               [7.65 / 100, 0.95 / 100, 0.71 / 100, 76.41 / 100, 14.28 / 100],
               [12.32 / 100, 5.96 / 100, 2.82 / 100, 2.16 / 100, 76.73 / 100]]  # LSTM

    CM_ConvLSTM = [[98.84 / 100, 0.15 / 100, 0.10 / 100, 0.11 / 100, 0.80 / 100],
               [18.12 / 100, 78.64 / 100, 0.09 / 100, 0.76 / 100, 2.38 / 100],
               [ 17.95  / 100, 0.30 / 100, 51.74 / 100, 0.18 / 100, 29.84 / 100],
               [14.77 / 100, 0.48 / 100, 0.13 / 100, 82.35 / 100, 2.27  / 100],
               [27.20 / 100, 2.76 / 100, 2.26 / 100, 1.35 / 100, 66.42 / 100]]  # LSTM

    output= 'D:/ConfusionMatrix0.png'
    xLabel = 'Predication of ConvLSTM'
    drawConfusionMatrix(CM_ConvLSTM,output,xLabel=xLabel)




if __name__=="__main__":
    #file_move('C:/DB/01-ARMSMOTN/2019/train/src','C:/DB/01-ARMSMOTN/2019/val/src', percent=0.2)
    #excel_path = r'C:\sj\a.xlsx'
    #readXLXSdatatoArr(excel_path, readcolds=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    #file_move('F:/01-NorthEast/02-CA/2017/train/src','F:/01-NorthEast/02-CA/2017/val/src', percent=0.2)
    '''a = Date2Julian('20210906')
    a = Julian2Date(2021,129)
    print(a)'''
    path = r'D:\05-GF2\12-test\04-MODISET'
    path2 = r'D:\05-GF2\12-test\04-MODISETPRJ'
    extention = '.tif'
    refilelist = getFileUnderPath(path2, extention)
    refilelist2 = []
    for m in refilelist:
        m = m.replace('04-MODISETPRJ','04-MODISET2')
        refilelist2.append(m)
    filelist = getFileUnderPath(path, extention)
    renameFileList(filelist, refilelist2)




