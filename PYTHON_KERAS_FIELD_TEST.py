import keras
from keras.preprocessing.image import img_to_array ,load_img

from mylib.PYTHON_LIBS_RASPBERRY_PI import ImageCropSet ,TimeCounter
from mylib.PYTHON_Keras_mAlexnet import MiniAlexnet

import os
import math
import numpy as np
import sys
import time
import gc
import firebase

import cv2

#FirebaseSendData Method
def FirebaseSendData(predict_set):
   ## Add data to firebase
    epoch_time_now = calendar.timegm(time.gmtime())
    for i ,val in predict_set.items():
        conv_val = np.asscalar(val)
        respond = firebase.put("/car-park-snapshots/swu/park0/p"+str(i), "val",conv_val )
        history_respond = firebase.put("/car-park-snapshots-history/swu/park0/p"+str(i)+"/"+str(epoch_time_now),"val", conv_val)

#Declare essential function
def ClearMemory(cameras,garbage_list):
    for i in range(len(garbage_list)):
        os.remove(garbage_list[i]) 
    for i , cam in enumerate(garbage_list):
        if cam.isOpened():
            cam.release()
    gc.collect()

#Initialize Files Directory
OS_UPPER_PATH = os.getcwd()
weights_file = OS_UPPER_PATH+"/KERAS_Weights_CNR_Parks_tuned_lr_1_e4_dc_5e_4_A.h5"
temp_img_name = "tmp_big_img_captured.jpg"
temp_cropped_img_name = "_tmp_cropped_img.jpg"
time_output_name = "time_statistics.txt"
garbage_list = []
time_counter = TimeCounter()
writer = FileWriterObject(OS_UPPER_PATH+"/"+time_output_name)

#Initialize firebase
firebase = firebase.FirebaseApplication('https://car-park-detection.firebaseio.com', None)

#Crop position set
img_crop_pos = {}
img_crop_pos[0] = ImageCropSet(x=458,y=518,w=60,h=60)
img_crop_pos[1] = ImageCropSet(x=385,y=480,w=67,h=56)
img_crop_pos[2] = ImageCropSet(x=646,y=504,w=60,h=60)
img_crop_pos[3] = ImageCropSet(x=149,y=310,w=16,h=16)
img_crop_pos[4] = ImageCropSet(x=133,y=301,w=15,h=15)
img_crop_pos[5] = ImageCropSet(x=458,y=518,w=60,h=60)
img_crop_pos[6] = ImageCropSet(x=385,y=480,w=67,h=56)
img_crop_pos[7] = ImageCropSet(x=646,y=504,w=60,h=60)
img_crop_pos[8] = ImageCropSet(x=149,y=310,w=16,h=16)
img_crop_pos[9] = ImageCropSet(x=133,y=301,w=15,h=15)
img_crop_pos[10] = ImageCropSet(x=149,y=310,w=16,h=16)
img_crop_pos[11] = ImageCropSet(x=133,y=301,w=15,h=15)

#Initialize Variables
batch_size = len(img_crop_pos)
image_color_channel = 3
image_size_w=224
image_size_h=224

#Initialize the camera
cameras = []
cam0 = cv2.VideoCapture(0)   # 0 -> index of camera
cam0_name = "cam0"
cam0_width = 320
cam0_height = 240
cam0.set(3,cam0_width)
cam0.set(4,cam0_height)
cameras.append(cameras)

# Allow the camera to warm up
time.sleep(0.1)

#Model initialize
model = MiniAlexnet(base_lr=0.0001,momentum=0.9,decay_rate=0.0005,nesterov=False)
model.load_weights(weights_file)
model.summary()

#Make sure the cam0 have been activation
if not (cam0.isOpened()):
    cam0.open()
    
#Get Image From Camera
return_boolean, captured_img = cam0.read()

if (return_boolean):
    cv2.imwrite(OS_UPPER_PATH+"/"+temp_img_name, captured_img)
    garbage_list.append(OS_UPPER_PATH+"/"+temp_img_name)
else:
    #Clear resources
    ClearMemory(cameras,garbage_list)
    sys.exit()
    
if (os.path.exists(OS_UPPER_PATH+"/"+temp_img_name)):
        for i, val in enumerate(img_crop_pos):
            #Crop specific position in each image
            img = Image.open(OS_UPPER_PATH+"/"+temp_img_name)
            cropped_image = img.crop((img_crop_pos[i].start_x,img_crop_pos[i].start_y,img_crop_pos[i].GetDestinationPosition(pos="x"),img_crop_pos[i].GetDestinationPosition(pos="y")))
            cropped_image.save(OS_UPPER_PATH+"/"+str(i)+temp_cropped_img_name)
            garbage_list.append(OS_UPPER_PATH+"/"+str(i)+temp_cropped_img_name)

x_tmp_list = []
for i in range(len(garbage_list)):
    cropped_img = load_img(garbage_list[i], target_size=(image_size_w, image_size_h))
    cropped_img = img_to_array(cropped_img)
    cropped_img = cropped_img.transpose((1,0,2))
    cropped_img /= 255
    x_tmp_list.append(cropped_img)
preproceed_images = np.array(x_tmp_list)

time_counter.StartCounting()
predicts =model.predict(preproceed_images,batch_size=batch_size)
time_counter.StopCounting()
writer.Write(text=time_counter.PrintElapsedTime())

package = []
for j,value in enumerate(predicts):
    predict = np.argmax(value)
    package.append(predict)

FirebaseSendData(package)

#Clear resources
ClearMemory(cameras,garbage_list)
sys.exit()
