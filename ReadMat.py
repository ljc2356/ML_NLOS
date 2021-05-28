import scipy.io as scio
import os
import numpy as np

def GetMat(data_folders):
    # data_folders = '.\data\20210413_indoor_AfterRecords'
    # data_folders = '.\\data\\20210413_indoor_AfterRecords'
    data_mat = []
    for root,dirs,files in os.walk(data_folders):
        print("读取到以下文件")
        for name in files:
            print(os.path.join(root,name))
            data_mat.append(scio.loadmat(os.path.join(root,name)))

    return data_mat

def Creat_Trainset(DatasetX,DatasetY,lookback=1):
    dataX,dataY = [],[]
    for i in range(len(DatasetX[:,0,0,0]) - lookback ):
        dataX.append(DatasetX[i:(i+lookback),:,:,:])
        dataY.append(DatasetY[i+lookback-1])   #对应DataX相应的最后一位
    npdataX = np.array(dataX)
    npdataY = np.array(dataY)
    return  npdataX,npdataY







