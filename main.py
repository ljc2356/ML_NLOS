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
# data_folders = './data/20210508_11201'
# dataset_folders = './data/Dataset'
# data_mat = GetMat(data_folders)
# dataX_Cir = []
# dataX_D = []
# dataX_Pdoa = []
# dataY_Target_loc = []
# dataX = []
# for i in range(len(data_mat)):
#     Observe_num = len(data_mat[i]['After_records'])
#     print("length of data = ", Observe_num)
#     Cir = np.zeros([Observe_num,8,50,1])
#     D = np.zeros([Observe_num,1])
#     Pdoa = np.zeros(([Observe_num,8]))
#     X = np.zeros([Observe_num,8,52,1])
#     Target_loc = np.zeros([Observe_num,2])
#     for k in range(Observe_num):
#         for antenna in range(8):
#             Cir[k,antenna,:,0] = (np.abs(data_mat[i]['After_records'][k,0]['uwbResult']['cir'][0,0][0,antenna])).reshape((1,50))
#             Cir[k, antenna, :,0] = Cir[k, antenna, :,0]/Cir[k, antenna, :,0].max()
#
#         X[k,:,0:50,0] = Cir[k,:,:,0]
#         Target_loc[k,:] = data_mat[i]['After_records'][k,0]['Target_loc'][0]
#         D[k,0] = data_mat[i]['After_records'][k,0]['meaResult']['D'][0][0][0][0]
#         Pdoa[k,0:8] = data_mat[i]['After_records'][k,0]['meaResult']['pdoa'][0][0][0]
#         X[k,:,50,0] = Pdoa[k,:]
#         X[k,:,51,0] = D[k,0]
#     if i == 0:
#         dataX_Cir = Cir
#         dataX_D = D
#         dataX_Pdoa = Pdoa
#         dataY_Target_loc = Target_loc
#         dataX = X
#     else:
#         dataX_Cir = np.vstack((dataX_Cir,Cir))
#         dataX_D = np.vstack((dataX_D,D))
#         dataX_Pdoa = np.vstack((dataX_Pdoa,Pdoa))
#         dataY_Target_loc = np.vstack((dataY_Target_loc,Target_loc))
#         dataX = np.vstack((dataX,X))
# dataY_D_Pdoa = D_AOA_cal(dataY_Target_loc)
# dataX_train,dataX_test,dataY_train,dataY_test = train_test_split(dataX,dataY_D_Pdoa,test_size=0.2,random_state=42)
#
# Dataset_dict = {'dataX':dataX,'dataY':dataY_D_Pdoa,'dataX_train':dataX_train,'dataX_test':dataX_test,'dataY_train':dataY_train,'dataY_test':dataY_test}
# savemat(file_name=os.path.join(dataset_folders,'Dataset'),mdict=Dataset_dict)

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
optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 1000000
latent_dim = 50
model = CVAE(latent_dim)
model.load_weights(filepath='./Model/')
rmse_mat = []
rmse_mat.append(loadmat('min_rmse.mat')['min_rmse'][0,0])

for epoch in range(1,epochs + 1):
    compute_apply_gradients(model,dataX_train,dataY_train,optimizer)
    # compute_rmse_apply_gradients(model, dataX_train, dataY_train, optimizer)
    loss = compute_loss(model,dataX_test,dataY_test)
    rmse = compute_rmse(model,dataX_test,dataY_test).numpy()

    if rmse <= min(rmse_mat):
        rmse_mat.append(rmse)
        mdic = {'min_rmse': rmse}
        savemat('min_rmse.mat', mdic)
        model.save_weights(filepath='./Model/')
        print('weights of model has been saved')

    print('Epoch:{},   Tese set loss:{},    Test set RMSE:{} '.format(epoch,loss,rmse))



