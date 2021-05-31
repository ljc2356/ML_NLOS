#%%
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from scipy.io import savemat
from scipy.io import loadmat
from ReadMat import *
from Myclass import *
from AOA import *
import time
#%%
data_folders = './data/20210508_11201'
dataset_folders = './data/Dataset'
dataset = loadmat(file_name=os.path.join(dataset_folders,'Dataset'))
dataX = dataset['dataX']
dataY = dataset['dataY']
dataX_train = dataset['dataX_train']
dataX_test = dataset['dataX_test']
dataY_train = dataset['dataY_train']
dataY_test = dataset['dataY_test']
#%%
latent_dim = 50
model = CVAE(latent_dim)
model.load_weights(filepath='./Model/')
rmse_D,rmse_PDOA = compute_rmse(model,dataX,dataY,split=True)
print('rmse of D:{} rmse of PDOA:{}'.format(rmse_D,rmse_PDOA))