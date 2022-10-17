#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2022/3/19 22:39
# @Author    :Lingbo Yang
# @Version   :v1.0
# @License: BSD 3 clause
# @Description: 建议使用mixAcc模式混合精度计算，传统float32模式相比混合模式，占用显存为mixAcc模式的155%，运算速度不同算法略有差异，UNET
#                 模型速度有轻微降低（小于1%），TFBS模型提高50%以上，ConvLSTM节约显存25%，速度提升44%

import glob
from torch.cuda.amp import autocast, GradScaler  # 用于混合精度运算
from utils.dataset import ISBI_Loader, ISBI_Loader_CV
from torch import optim
import torch.nn as nn
import torch
import math
from tqdm import tqdm
import numpy as np
import os
from utils.lucky import saveList2TXT_append


def train(train_loader, net, criterion, optimizer, epoch, epochs, device, scaler, sig_net, best_train_loss,
          best_train_loss_path, log_batch_path,log_epoch_path, it_step=20, eps=1e-10, mixAcc=True):
    # 切换模型为训练模式
    net.train()
    print('正在训练：第' + str(epoch) + '个epoch，共' + str(epochs) + '个。')
    # print("best loss:" + str(best_train_loss))
    int_now = 0
    # 用于计算精度,每次epoch前进行初始化
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    # 按照batch_size开始训练
    for image, label in tqdm(train_loader, position=0, leave=True):
        # print('正在训练第'+ str(int_now)+',共'+str(int_num)+'次')
        int_now += 1
        optimizer.zero_grad()

        # 是否使用混合精度模式进行训练，建议使用，节约显存，提高速度
        if mixAcc:
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.bfloat16)
            label = label.to(device=device, dtype=torch.bfloat16)
            with autocast():
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                # print('Loss/train', loss.item())
                # 保存loss值最小的网络参数
                if loss < best_train_loss:
                    best_train_loss = loss
                    # torch.save(net.state_dict(), 'best_train_model.pth')
                    torch.save(net.state_dict(), best_train_loss_path)
                    # print("best loss:" + str(best_loss))

            # Scales loss. 为了梯度放大.
            scaler.scale(loss).backward()
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)
            # 准备着，查看是否要增大scaler
            scaler.update()
        else:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            # print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_train_loss:
                best_train_loss = loss
                #torch.save(net.state_dict(), 'best_train_model.pth')
                torch.save(net.state_dict(), best_train_loss_path)
                # print("best loss:" + str(best_loss))

            # 更新参数
            loss.backward()
            optimizer.step()

        # 计算总体精度，召回率，F-score，Kappa系数
        #pred_sig = sig_net(pred)
        #pred_r = torch.round(pred_sig)
        #用desigmoid代替，提高效率，不用在pred的时候计算sigmoid函数
        #又因为0.5的de sigmoid其实就是0，所以只需要输出的值大于0，那么它就是1，否则就是0
        pred_r = (pred > 0).float()
        # 处理结果
        TP += (pred_r * label).sum()
        FP += ((1 - label) * pred_r).sum()
        FN += (label * (1 - pred_r)).sum()
        TN += ((1 - label) * (1 - pred_r)).sum()
        # print('TP:'+str(TP.data)+ '; FP:'+ str(FP.data)+'; FN:'+ str(FN.data)+ '; TN:'+ str(TN.data))
        # 每过10个it计算一次精度
        if int_now % it_step == 0:
            r = TP / (TP + FP + eps)
            p = TP / (TP + FN + eps)
            F1 = 2 * r * p / (r + p + eps)
            totalnum = TP + TN + FP + FN
            acc = (TP + TN) / (totalnum + eps)
            # print('r, p, f1, acc')
            # print(r, p, F1, acc)
            pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (totalnum * totalnum + eps)
            kappa = (acc - pe) / (1 - pe + eps)
            tqdm.write('batch_loss: {:.6f} best_loss: {:.6f} OA: {:.4f} f1: {:.4f} Kappa: {:.4f} recall: {:.4f} precision: {:.4f}'
                       .format(loss.data, best_train_loss.data, acc.data, F1.data, kappa.data,
                               r.data, p.data))
            batch_list=[epoch, int_now, best_train_loss.item(), acc.item(), F1.item(), kappa.item(),
                               r.item(), p.item()]
            saveList2TXT_append(batch_list,log_batch_path)
    #整个epoch计算一个最终的train loss
    r = TP / (TP + FP + eps)
    p = TP / (TP + FN + eps)
    F1 = 2 * r * p / (r + p + eps)
    totalnum = TP + TN + FP + FN
    acc = (TP + TN) / (totalnum + eps)
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (totalnum * totalnum + eps)
    kappa = (acc - pe) / (1 - pe + eps)
    tqdm.write(
        '第{:03d}个epoch best_loss: {:.6f} OA: {:.4f} f1: {:.4f} Kappa: {:.4f} recall: {:.4f} precision: {:.4f}'
        .format(epoch, best_train_loss.data, acc.data, F1.data, kappa.data,
                r.data, p.data))
    epoch_list = [epoch, best_train_loss.item(), acc.item(), F1.item(), kappa.item(),
                  r.item(), p.item()]
    saveList2TXT_append(epoch_list, log_epoch_path)

    return best_train_loss

def val(val_loader, net, criterion, epoch, epochs, device, sig_net, best_val_loss,
          best_val_loss_path, log_path, eps=1e-10, mixAcc=True,isSaveNet=False):
    '''
    验证模型
    :param val_loader:
    :param net:
    :param criterion:
    :param epoch:
    :param epochs:
    :param device:
    :param sig_net:
    :param best_val_loss:
    :param best_val_loss_path:
    :param log_path:
    :param eps:
    :param mixAcc:
    :param isSaveNet: 是否保存模型结构，如果是，那么导出文件中将包含模型结构和模型参数，否则只保存模型参数，结构还需自行定义
    :return:
    '''
    # 切换模型为训练模式
    net.eval()
    print('正在验证：第' + str(epoch) + '个epoch，共' + str(epochs) + '个。')
    # print("best loss:" + str(best_train_loss))
    # 用于计算精度,每次epoch前进行初始化
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    int_it = 0
    loss_total = 0
    bs_total=0
    # 按照batch_size开始训练
    for image, label in tqdm(val_loader):
        # print('正在训练第'+ str(int_now)+',共'+str(int_num)+'次')
        int_it += 1
        bs_n = label.shape[0]
        with torch.no_grad():
            # 是否使用混合精度模式进行验证，建议使用，节约显存，提高速度
            if mixAcc:
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.bfloat16)
                label = label.to(device=device, dtype=torch.bfloat16)
                with autocast():
                    # 使用网络参数，输出预测结果
                    pred = net(image)
                    # 计算loss
                    loss = criterion(pred, label)
                    loss_total = loss_total + loss
                    bs_total = bs_total + bs_n

            else:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                loss_total = loss_total + loss
                bs_total = bs_total + bs_n

            pred_r = (pred > 0).float()
            # 处理结果
            TP += (pred_r * label).sum()
            FP += ((1 - label) * pred_r).sum()
            FN += (label * (1 - pred_r)).sum()
            TN += ((1 - label) * (1 - pred_r)).sum()
    #整个epoch输出一个val精度即可
    r = TP / (TP + FP + eps)
    p = TP / (TP + FN + eps)
    F1 = 2 * r * p / (r + p + eps)
    totalnum = TP + TN + FP + FN
    acc = (TP + TN) / (totalnum + eps)
    # print('r, p, f1, acc')
    # print(r, p, F1, acc)
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (totalnum * totalnum + eps)
    kappa = (acc - pe) / (1 - pe + eps)
    loss_now = loss_total/bs_total
    if F1 > best_val_loss:
        best_val_loss = F1
        # torch.save(net.state_dict(), 'best_train_model.pth')
        if isSaveNet:
            torch.save(net, best_val_loss_path)
        else:
            torch.save(net.state_dict(), best_val_loss_path)
    print(
        '第{:03d}个epoch epoch_loss: {:.6f} OA: {:.4f} f1: {:.4f} Kappa: {:.4f} recall: {:.4f} precision: {:.4f}'
        .format(epoch, loss_now.data, acc.data, F1.data, kappa.data,
                r.data, p.data))
    epoch_list = [epoch, loss_now.item(), acc.item(), F1.item(), kappa.item(),
                  r.item(), p.item()]
    saveList2TXT_append(epoch_list, log_path)

    return best_val_loss

def getfile(data_path):
    imgs_path = glob.glob(os.path.join(data_path, 'src/*.tiff'))
    if os.path.exists(os.path.join(data_path, 'src/augmentation')):
        imgs_path.extend(glob.glob(os.path.join(data_path, 'src/augmentation/*.tiff')))
    return imgs_path


def train_net_cv(net, device, data_path, train_path, val_path, epochs=50, batch_size=16, lr=0.00001, best_weight_name = 'ConvLSTM_2017ZD',
              mixAcc=True, it_step=20,w_in=5):
    # 加载训练集
    # seed_torch()
    isbi_dataset = ISBI_Loader_CV(train_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    isbi_dataset_val = ISBI_Loader_CV(val_path)
    val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset_val,
                                               batch_size=batch_size*2,
                                               shuffle=True)
    #用来保存最佳训练损失权重的文件
    best_train_loss_path = os.path.join(data_path,
                               'model\\' + best_weight_name + '_best_train_loss_weight.pth')
    dir = os.path.dirname(best_train_loss_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    best_train_loss_path = best_train_loss_path.replace('/', '\\')

    # 用来保存最佳验证损失权重的文件
    best_val_loss_path = os.path.join(data_path,
                                        'model\\' + best_weight_name + '_best_val_loss_weight.pth')
    best_val_loss_path = best_val_loss_path.replace('/', '\\')

    #保存train batch精度的log文件
    train_batch_loss_log_path = os.path.join(data_path,
                                      'model\\' + best_weight_name + '_train_batch_loss.log')
    train_batch_loss_log_path = train_batch_loss_log_path.replace('/', '\\')
    head_log = ['epoch','batch','best_loss','OA','F1','Kappa','Recall','Precision']
    saveList2TXT_append(head_log,train_batch_loss_log_path)

    # 保存train 每个epoch精度的log文件
    train_epoch_loss_log_path = os.path.join(data_path,
                                             'model\\' + best_weight_name + '_train_epoch_loss.log')
    train_epoch_loss_log_path = train_epoch_loss_log_path.replace('/', '\\')
    head_log = ['epoch', 'best_loss', 'OA', 'F1', 'Kappa', 'Recall', 'Precision']
    saveList2TXT_append(head_log, train_epoch_loss_log_path)

    # 保存val 每个epoch精度的log文件
    val_epoch_loss_log_path = os.path.join(data_path,
                                             'model\\' + best_weight_name + '_val_epoch_loss.log')
    val_epoch_loss_log_path = val_epoch_loss_log_path.replace('/', '\\')
    head_log = ['epoch', 'epoch_loss', 'OA', 'F1', 'Kappa', 'Recall', 'Precision']
    saveList2TXT_append(head_log, val_epoch_loss_log_path)

    # 定义RMSprop算法
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), eps=1e-8, beta1=0.9, beta2=0.999)
    optimizer = optim.Adam(net.parameters())
    # 定义Loss算法 对于ConvLSTM，将w设置为7
    w = np.ones_like(1) * w_in

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(w, dtype=torch.float))
    # criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_train_loss = float('inf')
    best_val_loss = 0 #float('inf')

    # 单次训练次数
    len_img = len(isbi_dataset)
    int_num = math.ceil(len_img / batch_size)

    scaler = GradScaler()
    sig_net = nn.Sigmoid()
    eps = 1e-10
    import datetime
    starttime = datetime.datetime.now()
    # 训练epochs次
    for epoch in range(epochs):
        best_train_loss = train(train_loader, net, criterion, optimizer, epoch, epochs, device, scaler, sig_net,
              best_train_loss=best_train_loss, best_train_loss_path=best_train_loss_path,
                                log_batch_path= train_batch_loss_log_path,log_epoch_path= train_epoch_loss_log_path,
                                it_step=it_step, eps=eps, mixAcc=mixAcc)
        best_val_loss = val(val_loader, net, criterion, epoch, epochs, device, sig_net, best_val_loss=best_val_loss,
          best_val_loss_path=best_val_loss_path, log_path=val_epoch_loss_log_path, eps=eps, mixAcc=mixAcc)
        # 输出整个epoch的精度
        #print('在这里输出整个epoch的精度，包括训练精度和验证精度，只有在验证精度提高的情况下才将本次训练的参数传递下去，否则保持原样')
    endtime = datetime.datetime.now()
    print('共耗时：'+str(endtime - starttime))
    head_log = ['共耗时：', str(endtime - starttime)]
    saveList2TXT_append(head_log, val_epoch_loss_log_path)
    saveList2TXT_append(head_log, train_batch_loss_log_path)
    saveList2TXT_append(head_log, train_epoch_loss_log_path)



def trainCV_ATTTFBS_MAIN(mixAcc=True):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    # 加载网络，图片单通道1，分类为1。
    # net = UNet32(n_channels=18, n_classes=1)
    bs = 16
    #net = TFBS(bs=bs, n_laryers=1, hiddensize=64)

    # 指定训练集地址，开始训练
    data_path = "F:/ARMSMOTN/2017/"
    img_path = getfile(data_path)
    from sklearn.model_selection import KFold
    splitnum = 10
    seed = 41
    kf = KFold(n_splits=splitnum, shuffle=True, random_state=seed)
    cv_nb = 0
    nb_repeat = 10
    for train_index, test_index in kf.split(img_path):
        #把net放这里，不然会不重置
        from model.TFBS_ATT_model import TFBS_ATT
        net = TFBS_ATT(bs=bs, n_laryers=1, hiddensize=64)
        # 将网络拷贝到deivce中
        net.to(device=device)
        train_set = np.array(img_path)
        train_split_set, test_split_set = train_set[train_index], train_set[test_index]
        train_split_set = np.tile(train_split_set, nb_repeat)
        train_split_set = train_split_set.tolist()
        test_split_set = test_split_set.tolist()
        train_net_cv(net, device, data_path, train_split_set, test_split_set,
                     best_weight_name='AttTFBS-2019-cv' + str(cv_nb) + '-50Epoch-bs16-nl1-hs64-unet64-noAug'
                     , batch_size=bs, mixAcc=mixAcc, epochs=50, w_in=1)
        cv_nb = cv_nb + 1


if __name__ == "__main__":
    print('Start...')
    trainCV_ATTTFBS_MAIN()
    print('Done！')
