import keras
from keras.callbacks import ModelCheckpoint

from mylib.PYTHON_LIB_myclasses import FileWriterObject,ConfusionMatrixObject,PlotLearning,PlotLosses
from mylib.PYTHON_Keras_mAlexnet import MiniAlexnet
from mylib.PYTHON_tools import read_labels

import os
import math

#Initialize Files Directory
OS_UPPER_PATH = os.getcwd()
train_labels_path = OS_UPPER_PATH+"/Resources/NewModels/splits/CNRParkAB/train.txt"
val_labels_path = OS_UPPER_PATH+"/Resources/NewModels/splits/CNRPark-EXT/val.txt"
train_root_images_folder = OS_UPPER_PATH+"/Resources/CNR_PARK"
val_root_images_folder = OS_UPPER_PATH+"/Resources/CNR_PARK_EXT/PATCHES"
weight_output_filename = "KERAS_Weights_CNR_Parks_tuned_lr4dc1.h5"
checkpoints_folder = OS_UPPER_PATH+"/Checkpoints"
checkpoint_filename = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

#Initialize Variables
batch_size = 64
image_color_channel = 3
image_size_w=224
image_size_h=224
resume_at_epoch=0

model = MiniAlexnet(base_lr=0.0004,momentum=0.9,decay_rate=0.0001,nesterov=False)
model.summary()
        
x_train,y_train,train_num = read_labels(train_labels_path,(256,256),(224,224),train_root_images_folder,batch_size=batch_size,initial_epoch=resume_at_epoch,random_horizontal_flip=True)
x_val,y_val,val_num = read_labels(val_labels_path,(224,224),(224,224),val_root_images_folder,batch_size=batch_size,initial_epoch=resume_at_epoch,random_horizontal_flip=True)

print("Read images complete.\n")

callback0= ModelCheckpoint(checkpoints_folder+"/(PARAMS);TRAIN_NUM="+str(train_num)+";BATCH_SIZE="+str(batch_size)+"||"+checkpoint_filename, monitor='val_acc', verbose=1, save_best_only=False,period=5000)
#callback1= PlotLosses()
#callbacks_list = [callback0,callback1]
callbacks_list = [callback0]

print("Ready to fit the model.")

model.fit(x=x_train
          ,y=y_train
          ,batch_size=batch_size
          ,epochs=math.ceil(train_num/batch_size)
          ,verbose=2
          ,callbacks=callbacks_list
          ,initial_epoch=resume_at_epoch
         ,shuffle=False
         ,validation_data=(x_val,y_val))
model.save_weights(weight_output_filename)

print("Train the model complete.")