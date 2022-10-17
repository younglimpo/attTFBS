# attTFBS model
The source code of attTFBS model

## Introduction

The attTFBS model contains three modules: an attention-based LSTM (attLSTM) module used for extracting deep-seated temporal features, UNET module used for extracting spatial semantic information from the temporal features received from the attLSTM module, and output module to perform classification based on the features fed by the UNET module

The architecture of the attTFBS model is shown below:
![Image text](https://github.com/younglimpo/TFBSmodel/blob/master/TFBS%20architecture.png)

## Environment

| Library | Version | 
| :-----:| :----: | 
| Python | 3.7.0 | 
| Pytorch | 1.11.0 | 
| scikit-learn | 0.20.3 | 
| gdal | 2.3.3 | 
| scikit-learn | 1.0.2 | 

## Dataset 

The training dataset are shared by google drive: 
https://drive.google.com/drive/folders/120X2tLv4-6pxIREOMFFGILId4R98gdWK?usp=sharing

The dataset is generated from time-series Sentinel-1 SAR images in 2019 in AR,MS, MO, TN of the United States, and Cropland Data Layer (CDL) is used as the label data.

The time-series Sentinel-1 SAR images is preprocessed and downloaded by Google Earth Engine and the linke of the code can be found below:
https://code.earthengine.google.com/49f8e2532075272a79883ad8fbf41ccb

Download two compressed files named 'src' and 'label' to your local computer and unzip them to the same directory.
![Image text](https://github.com/younglimpo/TFBSmodel/blob/master/Img/dataset.png)

Each image tile in the src folder contains 18 channels with a spatial size of 128 × 128.


## User Guide

Open train.py, change the 'data_path' parameter to the directory where the src and label folers is.
Configure the bs (batch size) according to the memory size of the vedio card on your compurter.
The specific parameters of the attTFBS model can be found in the 'TFBS_ATT()' function.
The result of the cross-validation will be printed in the default log file.

 ```python
 def trainCV_ATTTFBS_MAIN(mixAcc=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    bs = 16
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

```

 ## Results
 ![Loss](https://github.com/younglimpo/TFBSmodel/blob/master/Img/loss.png) ![Overall accuracy](https://github.com/younglimpo/TFBSmodel/blob/master/Img/OA.png) ![F-score](https://github.com/younglimpo/TFBSmodel/blob/master/Img/f-score.png)![Kappa](https://github.com/younglimpo/TFBSmodel/blob/master/Img/kappa.png) ![Recall](https://github.com/younglimpo/TFBSmodel/blob/master/Img/recall.png) ![Precision](https://github.com/younglimpo/TFBSmodel/blob/master/Img/precision.png)
 
