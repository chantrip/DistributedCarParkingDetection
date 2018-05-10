import keras
from keras.callbacks import ModelCheckpoint

from mylib.PYTHON_LIB_myclasses import FileWriterObject,ConfusionMatrixObject,PlotLearning,PlotLosses
from mylib.PYTHON_Keras_mAlexnet import MiniAlexnet
from mylib.PYTHON_Keras_mAlexnet import MiniAlexnetTrainProperties as matp
from mylib.PYTHON_tools import read_labels_batch_out

import os
import math
import numpy as np

#Initialize Files Directory
model_prop = matp.GetTrainSettingsValue(index=1)

OS_UPPER_PATH = os.getcwd()
labels_path = OS_UPPER_PATH+"/Resources/SWUPark/test.txt"
root_images_folder = OS_UPPER_PATH+"/Resources/SWUPark/All"
weights_file = OS_UPPER_PATH+"/"+model_prop.weight_filename
text_output_filename = "TEXT_"+model_prop.weight_filename

#Initialize Variables
batch_size = 1000
image_color_channel = 3
image_size_w=224
image_size_h=224
resume_at_epoch=0

#Initialize Necessary  Object
confusion_mat = ConfusionMatrixObject()
file_writer = FileWriterObject(f_name=OS_UPPER_PATH+"/"+text_output_filename+".final")

model = MiniAlexnet(base_lr=model_prop.base_lr,momentum=model_prop.momentum,decay_rate=model_prop.decay_rate,nesterov=model_prop.nesterov)
model.load_weights(weights_file)
model.summary()

confusion_mat = ConfusionMatrixObject()

i=0
while(True):
    x_test,y_test,test_num = read_labels_batch_out(labels_path,(224,224),(224,224),root_images_folder,batch_size=batch_size,iteration_num=i,random_horizontal_flip=False)
    if (test_num ==0):
        break;
        
    predicts =model.predict(x_test,batch_size=batch_size)
    
    for j,value in enumerate(predicts):
        predict = np.argmax(value)
        truth_val = y_test[j]
        confusion_mat.AddValueToConfusionMat(val=predict,label_val=truth_val)
        file_writer.Write("[["+str(j+(batch_size*i))+"]] "+str(truth_val)+" {"+str(predict)+"}")
        
    print("====================================================================")
    file_writer.Write("====================================================================")

    print("result of iteration number : "+str(i))
    file_writer.Write("result of iteration number : "+str(i))
    print(" ")
    file_writer.Write(" ")

    print("---Show Current Score---\n")
    file_writer.Write("---Show Current Score---\n")
    print(confusion_mat.PrintConfusionMat())
    file_writer.Write(confusion_mat.PrintConfusionMat())
    print(confusion_mat.PrintAccuracy())
    file_writer.Write(confusion_mat.PrintAccuracy())
    print(confusion_mat.PrintPrecision())
    file_writer.Write(confusion_mat.PrintPrecision())
    print(confusion_mat.PrintRecall())
    file_writer.Write(confusion_mat.PrintRecall())
    print(confusion_mat.PrintF1Score())
    file_writer.Write(confusion_mat.PrintF1Score())

    print("====================================================================")
    file_writer.Write("====================================================================")
        
    i=i+1
        
print("====================================================================")
file_writer.Write("====================================================================")
print("---Test Completed---")
file_writer.Write("---Test Completed---")
print("====================================================================")
file_writer.Write("====================================================================")
print("---Summary---")
file_writer.Write("---Summary---")
print("TESTED IMAGES COUNT : "+str(test_num))
file_writer.Write("TESTED IMAGES COUNT : "+str(test_num))
print("====================================================================")
file_writer.Write("====================================================================")
print("====================================================================")
file_writer.Write("====================================================================")
print("---Show Final Score---\n")
file_writer.Write("---Show Final Score---\n")
print(confusion_mat.PrintConfusionMat())
file_writer.Write(confusion_mat.PrintConfusionMat())
print(confusion_mat.PrintAccuracy())
file_writer.Write(confusion_mat.PrintAccuracy())
print(confusion_mat.PrintPrecision())
file_writer.Write(confusion_mat.PrintPrecision())
print(confusion_mat.PrintRecall())
file_writer.Write(confusion_mat.PrintRecall())
print(confusion_mat.PrintF1Score())
file_writer.Write(confusion_mat.PrintF1Score())
print("---End---")