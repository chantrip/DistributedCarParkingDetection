import keras
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation , Convolution2D,MaxPooling2D,Flatten,ZeroPadding2D
from keras.constraints import *

def l1_reg(multiplier,weight_matrix):
    return multiplier * K.sum(K.abs(weight_matrix))

def l2_reg(multiplier,weight_matrix):
    return multiplier * K.square(K.abs(weight_matrix))
    
def MiniAlexnet(base_lr=0.0008,momentum=0.9,decay_rate=0.0001,nesterov=False):
    model = Sequential()
    input_shape = (224,224,3)
    model.add(Convolution2D(filters=16, kernel_size=(11,11),strides=(4,4), input_shape=input_shape,
                           padding='valid',activation='relu',use_bias=True
                            ,kernel_initializer='glorot_uniform',bias_initializer='ones'))
                           #,kernel_regularizer=regularizers.l1_l2(1/5,1/5),bias_regularizer=regularizers.l1_l2(2/5,0/5)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))

    model.add(Convolution2D(filters=20, kernel_size=(5,5),strides=(1,1),
                           padding='valid',activation='relu',use_bias=True
                            ,kernel_initializer='glorot_uniform',bias_initializer='ones'))
                           #,kernel_regularizer=regularizers.l1_l2(1/5,1/5),bias_regularizer=regularizers.l1_l2(2/5,0/5)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Convolution2D(filters=30, kernel_size=(3,3),strides=(1,1),
                           padding='valid',activation='relu',use_bias=True
                            ,kernel_initializer='glorot_uniform',bias_initializer='ones'))
                           #,kernel_regularizer=regularizers.l1_l2(1/5,1/5),bias_regularizer=regularizers.l1_l2(2/5,0/5)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(48,
                   activation='relu',use_bias=True
                   ,kernel_initializer='glorot_uniform',bias_initializer='ones'))
                   #,kernel_regularizer=regularizers.l1_l2(1/5,1/5),bias_regularizer=regularizers.l1_l2(2/5,1/2)))
    model.add(Dense(2,
                   activation='softmax',use_bias=True
                   ,kernel_initializer='glorot_uniform',bias_initializer='ones'))
                   #,kernel_regularizer=regularizers.l1_l2(1/5,1/5),bias_regularizer=regularizers.l1_l2(2/5,1/2)))

    model.compile(optimizer=optimizers.SGD(lr=base_lr, momentum=momentum, decay=decay_rate, nesterov=nesterov),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class MiniAlexnetTrainProperties:
    base_lr = 0
    momentum = 0
    decay_rate = 0
    nesterov = False
    weight_filename = ""
    
    def __init__(self,*args, **kwargs):
        self.base_lr = args[0]
        self.momentum = args[1]
        self.decay_rate = args[2]
        self.nesterov = args[3]
        self.weight_filename = args[4]

    @classmethod
    def GetTrainSettingsValue(cls,index):
        args=[]
        if index ==0:
            args = [0.0001,0.9,0.0001,False,"KERAS_Weights_CNR_Parks_tuned_lr_1_e4_dc_1e_4_C.h5"]
        elif index==1:
            args = [0.0001,0.9,0.0005,False,"KERAS_Weights_CNR_Parks_tuned_lr_1_e4_dc_5e_4_C.h5"]
        elif index==2:
            args = [0.0004,0.9,0.0001,False,"KERAS_Weights_CNR_Parks_tuned_lr_4_e4_dc_1e_4_C.h5"]
        elif index==3:
            args = [0.0004,0.9,0.0005,False,"KERAS_Weights_CNR_Parks_tuned_lr_4_e4_dc_5e_4_C.h5"]
        elif index==4:
            args = [0.0008,0.9,0.0001,False,"KERAS_Weights_CNR_Parks_tuned_lr_8_e4_dc_1e_4_C.h5"]
        elif index==5:
            args = [0.0008,0.9,0.0005,False,"KERAS_Weights_CNR_Parks_tuned_lr_8_e4_dc_5e_4_C.h5"]

        return cls(*args)