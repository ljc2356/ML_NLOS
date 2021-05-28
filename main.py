#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

from ReadMat import *
from Myclass import Autoencoder
from Myclass import NLOS_Model_List
from AOA import *

#%%
data_folders = './data/20210508_11201'
data_mat = GetMat(data_folders)
dataX_Cir = []
dataX_D_Pdoa = []
dataY_Target_loc = []
for i in range(len(data_mat)):
    Observe_num = len(data_mat[i]['After_records'])
    print("length of data = ", Observe_num)
    Cir = np.zeros([Observe_num,8,50,1])
    D_pdoa = np.zeros([Observe_num,1+8])
    Target_loc = np.zeros([Observe_num,2])
    for k in range(Observe_num):
        for antenna in range(8):
            Cir[k,antenna,:,0] = (np.abs(data_mat[i]['After_records'][k,0]['uwbResult']['cir'][0,0][0,antenna])).reshape((1,50))
            Cir[k, antenna, :,0] = Cir[k, antenna, :,0]/Cir[k, antenna, :,0].max()
        Target_loc[k,:] = data_mat[i]['After_records'][k,0]['Target_loc'][0]
        D_pdoa[k, 0] = data_mat[i]['After_records'][k,0]['meaResult']['D'][0][0][0][0]
        D_pdoa[k, 1:9] = data_mat[i]['After_records'][k,0]['meaResult']['pdoa'][0][0][0]
    if i == 0:
        dataX_Cir = Cir
        dataX_D_Pdoa = D_pdoa
        dataY_Target_loc = Target_loc
    else:
        dataX_Cir = np.vstack((dataX_Cir,Cir))
        dataX_D_Pdoa = np.vstack((dataX_D_Pdoa,D_pdoa))
        dataY_Target_loc = np.vstack((dataY_Target_loc,Target_loc))

dataX_Cir_train,dataX_Cir_test =train_test_split(dataX_Cir,test_size=0.2,random_state= 1)
#%% autoencoder difination
latent_dim = 10
# EncoderDecoder = Autoencoder(latent_dim)
AE_filepath = 'path_model'
# EncoderDecoder = keras.models.load_model(AE_filepath)
# EncoderDecoder.compile(optimizer='adam',loss = keras.losses.mean_squared_error)
# Reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
# EncoderDecoder.fit(dataX_Cir_train,dataX_Cir_train,epochs=100,shuffle=True,callbacks=Reduce_lr,validation_data=(dataX_Cir_test,dataX_Cir_test))
# EncoderDecoder.save(AE_filepath)
#%% load module and cluster

EncoderDecoder = keras.models.load_model(AE_filepath)
Encoded_dataX_Cir =np.array(EncoderDecoder.encoder(dataX_Cir))
clustering = DBSCAN(eps=10,min_samples=2)
Encoded_dataX_Cir_labels  = clustering.fit_predict(Encoded_dataX_Cir).reshape(-1,1)

dataX = np.hstack((dataX_D_Pdoa,Encoded_dataX_Cir))
dataY = D_AOA_cal(dataY_Target_loc)
dataX_ExpendLabels = np.hstack((Encoded_dataX_Cir_labels,dataX))  # the first collum is the index of dataX
dataX_ExpendLabels_train,dataX_ExpendLabels_test,dataY_train,dataY_test = train_test_split(
    dataX_ExpendLabels,dataY,test_size=0.1,random_state=1
)

#%% try to expend model and train
Multi_Model_folderpath = './Model/'
Num_NN = int(max(dataX_ExpendLabels_train[:,0])) + 1
Multi_Model = NLOS_Model_List(Num_NN)
Multi_Model.compile(optimizer='adam',loss='mean_squared_error')
Multi_Model(dataX_ExpendLabels_train)
Multi_Model.load_weight(Multi_Model_folderpath)

Reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
Early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.000001,patience=200)

Multi_Model.fit(dataX_ExpendLabels_train,dataY_train,epochs=10000,validation_split=0.2,shuffle=True,callbacks=(Reduce_lr,Early_stop))
Multi_Model.save_weight(folderpath=Multi_Model_folderpath)

data_Predict_test = Multi_Model(dataX_ExpendLabels_test)

