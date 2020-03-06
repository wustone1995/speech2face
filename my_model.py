import tensorflow as tf
import pydot
import graphviz

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def AudioOnlyModel(audio_shape = [598,257]):

    ip = tf.keras.layers.Input(shape =(audio_shape[0],audio_shape[1],2))

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(ip)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    #check 2-1 ratio
    x = tf.keras.layers.MaxPool2D( pool_size=[2,1], strides=(2,1))(x)
    
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(6,1),strides=1,padding="VALID")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    flatten = tf.keras.layers.Flatten()(x)
    
    dense = tf.keras.layers.Dense(4096, activation = "relu")(flatten)
    dense = tf.keras.layers.Dense(4096)(dense)

    model = tf.keras.Model(ip, dense)
    return model



from tensorflow.keras.utils import plot_model

my_model = AudioOnlyModel()

def model_summary(model_vars = my_model):
    model_vars.summary()

# model_summary()

# batchsize = 3
import numpy as np

x_train = np.random.rand(10,598,257,2)
y_train = np.random.rand(10,4096)


x = x_train[-1:]
y_predicted = my_model.predict(x)
print("last max,min",np.amax(x_train),np.amin(x_train),np.amax(y_train),np.amin(y_train))


_lambda=1
epsilon=1e-12
def normalize(x):
    return x / np.sqrt(max(np.sum(x**2), epsilon))
def my_loss_funct2(y_true, y_pred):
        return _lambda *np.sum(( normalize(y_true)-normalize(y_pred))**2)

print("current loss",my_loss_funct2(y_predicted,y_train[-1:]))

def loss_helper():
    def my_loss_funct(y_true, y_pred):
        return _lambda *( tf.math.reduce_sum((tf.math.l2_normalize(y_true)-tf.math.l2_normalize(y_pred))**2 ))
    return my_loss_funct

my_loss = loss_helper()
my_opt = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# my_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5,decay=0.95, amsgrad=False)
# my_model.compile(optimizer=my_opt,loss='mean_squared_error')
my_model.compile(optimizer=my_opt,loss=my_loss)
my_model.fit(x=x_train,y=y_train,epochs=100,batch_size=1,verbose=1,initial_epoch=0)


x = x_train[-1:]
y_predicted = my_model.predict(x)
# print(np.sum(np.abs(y_predicted-y_train[-1:])**2))
print("current loss",my_loss_funct2(y_predicted,y_train[-1:]))

print("last max,min",np.amax(x_train),np.amin(x_train),np.amax(y_train),np.amin(y_train))

