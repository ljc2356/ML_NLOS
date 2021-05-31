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



class CVAE(tf.keras.Model):
    def __init__(self,latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(8,52,1)),
            tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,activation='relu',padding="SAME"),
            tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=2,activation='relu',padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)   #no activation
        ])
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=latent_dim),
            tf.keras.layers.Dense(units=2*13*64,activation="relu"),
            tf.keras.layers.Reshape(target_shape=(2,13,64)),
            tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(2,2),padding="SAME",activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=(2,2),padding="SAME",activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=8*52*2)  #no activation
        ])
        self.MLP_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=latent_dim),
            tf.keras.layers.Dense(units=latent_dim * 4,activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=latent_dim * 8,activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=9),
            tf.keras.layers.LeakyReLU()
        ])

    def encode(self,x):
        mean_zx,logvar_zx = tf.split(self.inference_net(x),num_or_size_splits=2,axis=1)
        return mean_zx,logvar_zx
    def SampleAndGenZ(self,mean,logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    def decode(self,z):
        mean_xz,logvar_xz = tf.split(self.generative_net(z),num_or_size_splits=2,axis=1)
        return mean_xz,logvar_xz
    def SampleAndGenX(self,mean,logvar):
        eps = tf.random.normal(shape=mean.shape)
        Flatten_x = eps * tf.exp(logvar * 0.5) + mean
        return tf.reshape(tensor=Flatten_x,shape=[8,-1])
    def Predict(self,z):
        pre_D_PDOA = self.MLP_net(z)
        return pre_D_PDOA
    def call(self, inputs, training=None, mask=None):
        mean_zx,logvar_zx = self.encode(inputs)
        z = self.SampleAndGenZ(mean_zx,logvar_zx)
        pre_y = self.MLP_net(z)
        return pre_y


def log_norm_pdf(sample,mean,logvar,raxis = 1):
    mean = tf.cast(mean,tf.float64)
    logvar = tf.cast(logvar,tf.float64)
    sample = tf.cast(sample,tf.float64)
    log2pi = tf.cast(tf.math.log(2* np.pi),tf.float64)
    prob = tf.reduce_sum(
        -0.5 * ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi)
    )
    return prob


def compute_loss(model,x,y):
    mean_zx,logvar_zx = model.encode(x)
    z = model.SampleAndGenZ(mean_zx,logvar_zx)
    mean_xz,logvar_xz = model.decode(z)

    x_flatten = x.reshape(x.shape[0],8 * 52)
    logpx_z = log_norm_pdf(x_flatten,mean_xz,logvar_xz)
    logpz = log_norm_pdf(z,0.,0.)
    logpz_x = log_norm_pdf(z,mean_zx,logvar_zx)
    VAE_loss = -1 * tf.reduce_mean(logpx_z + logpz - logpz_x)

    pre_y = model.MLP_net(z)
    rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pre_y - y)**2,axis=1)))
    rmse = tf.cast(rmse,VAE_loss.dtype)

    sum_loss = VAE_loss + rmse
    return sum_loss

def compute_rmse(model,x,y,split = False):
    mean_zx, logvar_zx = model.encode(x)
    z = model.SampleAndGenZ(mean_zx, logvar_zx)
    pre_y = model.MLP_net(z)
    if split == False:
        rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pre_y - y)**2,axis=1)))
        return rmse
    else:
        rmse_D = tf.sqrt(tf.reduce_mean((pre_y[:,0] - y[:,0])**2))
        rmse_PDOA = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pre_y[:,1:] - y[:,1:])**2,axis=1)))
        return rmse_D,rmse_PDOA

def compute_apply_gradients(model,x,y,optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model,x,y)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

def compute_rmse_apply_gradients(model,x,y,optimizer):
    with tf.GradientTape() as tape:
        rmse = compute_rmse(model,x,y)
    gradients = tape.gradient(rmse,model.MLP_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.MLP_net.trainable_variables))







