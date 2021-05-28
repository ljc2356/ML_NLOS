import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class Autoencoder(keras.Model):
    def __init__(self,latent_dim):
        super(Autoencoder,self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=[8,50,1]),
            keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',strides=2),
            keras.layers.Conv2D(8,(3,3),activation='relu',padding='same',strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim,activation='relu')])
        self.decoder = keras.Sequential([    #this dimension is designed
            keras.layers.InputLayer(input_shape=[latent_dim]),
            keras.layers.Dense(208,activation='relu'),
            keras.layers.Reshape([2,13,8]),
            keras.layers.Conv2DTranspose(8,kernel_size=3,strides=2,activation='relu',padding='same'),
            keras.layers.Conv2DTranspose(16,kernel_size=3,strides=2,activation='relu',padding='same'),
            keras.layers.Conv2D(1,kernel_size=(1,3),activation='sigmoid'),
        ])
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class NN_cell(keras.Model):
    def __init__(self):
        super(NN_cell, self).__init__()
        self.NN = keras.Sequential([
            keras.layers.InputLayer(input_shape=[19]),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(9),
            keras.layers.LeakyReLU()
        ])
    def call(self,x):
        output = self.NN(x)
        return output

class NLOS_Model_List:
    def __init__(self,num_NN):
        self.modellist = []
        for i in range(num_NN):
            self.modellist.append(NN_cell())
    def __call__(self, dataX):
        max_dataX_index = int(max(dataX[:,0]))
        min_dataX_index = int(min(dataX[:,0]))
        Ylist = np.zeros(shape=[dataX.shape[0],9])
        for i in range(min_dataX_index,max_dataX_index+1):
            ith_index = (dataX[:,0]==i)
            Pre_result = self.modellist[i](dataX[ith_index , 1:])
            Ylist[ith_index,:] = Pre_result
        return Ylist
    def compile(self,optimizer = 'rmsprop',loss =None):
        for i in range(len(self.modellist)):
            self.modellist[i].compile(optimizer=optimizer,loss=loss)
        return True
    def fit(self,dataX,dataY,epochs = 1,validation_split = 0.0,shuffle = True,callbacks = None):
        max_dataX_index = int(max(dataX[:,0]))
        min_dataX_index = int(min(dataX[:,0]))
        for i in range(min_dataX_index,max_dataX_index+1):
            ith_index = (dataX[:,0] == i)
            self.modellist[i].fit(dataX[ith_index,1:],dataY[ith_index,:],epochs=epochs,validation_split=validation_split,shuffle=True,callbacks=callbacks)
        return True

    def save_weight(self,folderpath,overwrite = True):
        for i in range(len(self.modellist)):
            modelname = '%d.h5' %i
            FullModelName = folderpath + modelname
            self.modellist[i].save_weights(filepath=FullModelName,overwrite=overwrite)

    def load_weight(self,folderpath):
        for i in range(len(self.modellist)):
            modelname = '%d.h5' %i
            FullModelName = folderpath + modelname
            self.modellist[i].load_weights(filepath= FullModelName)











