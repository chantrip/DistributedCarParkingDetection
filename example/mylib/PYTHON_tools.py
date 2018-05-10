import numpy as np
import os
from keras.preprocessing.image import img_to_array ,load_img
from PIL import Image
import random
import math

import datetime

def random_crop(img,original_size, cropped_size):
    if original_size <= cropped_size :
        return img
    orig_w , orig_h = original_size
    crop_w , crop_h = cropped_size
    max_w_can_rand = orig_w - crop_w
    max_h_can_rand = orig_h - crop_h
    rand_w = random.randint(0, max_w_can_rand)
    rand_h = random.randint(0, max_h_can_rand)
    cropping = ((rand_h, rand_h+crop_h), (rand_w, rand_w+crop_w))
    return img[cropping[0][0]:cropping[0][1], cropping[1][0]:cropping[1][1], :] 

def read_labels(labels_path,size,crop_size,root_images_folder,batch_size,initial_epoch,random_horizontal_flip): 
    all_total_time = 0
    n = 0
    all_n=0
    all_n_subtract =0
    w_size,h_size = size
    crop_w_size , crop_h_size = crop_size
    x_tmp_list = []
    y_tmp_list = []
    resume_data_at = initial_epoch*batch_size
    with open(labels_path) as f_counter:
        for line in f_counter:
            if (all_n_subtract < resume_data_at):
                all_n_subtract = all_n_subtract + 1
            all_n = all_n + 1
    
    all_n = all_n - all_n_subtract
    if (all_n <=0):
        return None,None,0
    
    with open(labels_path) as f:
        for line in f:
            if (n < resume_data_at):
                n = n + 1
                continue
            start_time = datetime.datetime.now()
            
            #Get Image Directory String From The File
            txt_val = (root_images_folder+"/"+line.rstrip('\n')).split()

            #Declare Variables
            image_dir = txt_val[0]
            image_label = int(txt_val[1])
            
            #Preprocessing
            img = load_img(image_dir, target_size=(w_size, h_size))
            #50 50 
            if (random.randint(0, 1) == 0 and random_horizontal_flip== True):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img_to_array(img)
            #print(img.shape)
            img = random_crop(img,original_size=(w_size,h_size),cropped_size=(crop_w_size,crop_h_size))
            #print(img.shape)
            img = img.transpose((1,0,2))
            img /= 255
            
 
            #print(x_numpy.shape)
            #img = np.expand_dims(img, axis=0)
            #print("#"+str(img.shape))
            x_tmp_list.append(img)
            y_tmp_list.append(image_label)    
            n = n + 1
            
            end_time = datetime.datetime.now()
            total_time = (end_time-start_time).total_seconds()
            all_total_time = all_total_time+total_time
            print("Load images progress : "+str(n-all_n_subtract)+" / "+str(all_n)+" , Remaining Time : "
                  +str(datetime.timedelta(seconds=total_time*(all_n-(n-all_n_subtract)))).split('.')[0]
                  +" , Total Time : "+str(datetime.timedelta(seconds=all_total_time)).split('.')[0], end="\r")
            
    print("Load images complete \n")
    return np.array(x_tmp_list),np.array(y_tmp_list),n

def read_labels_batch_out(labels_path,size,crop_size,root_images_folder,batch_size,iteration_num,random_horizontal_flip): 
    all_total_time = 0
    n = 0
    all_n=0
    all_n_subtract =0
    w_size,h_size = size
    crop_w_size , crop_h_size = crop_size
    x_tmp_list = []
    y_tmp_list = []
    resume_data_at = iteration_num*batch_size
    limit_count = 0
    with open(labels_path) as f_counter:
        for line in f_counter:
            if (all_n_subtract < resume_data_at):
                all_n_subtract = all_n_subtract + 1
            all_n = all_n + 1
    
    all_n = all_n - all_n_subtract
    if (all_n <=0):
        return None,None,0
    
    with open(labels_path) as f:
        for line in f:
            if (n < resume_data_at):
                n = n + 1
                continue
            
            if (limit_count >=batch_size):
                break
                
            start_time = datetime.datetime.now()
            
            #Get Image Directory String From The File
            txt_val = (root_images_folder+"/"+line.rstrip('\n')).split()

            #Declare Variables
            image_dir = txt_val[0]
            image_label = int(txt_val[1])
            
            #Preprocessing
            img = load_img(image_dir, target_size=(w_size, h_size))
            #50 50 
            if (random.randint(0, 1) == 0 and random_horizontal_flip== True):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img_to_array(img)
            #print(img.shape)
            img = random_crop(img,original_size=(w_size,h_size),cropped_size=(crop_w_size,crop_h_size))
            #print(img.shape)
            img = img.transpose((1,0,2))
            img /= 255
 
            #print(x_numpy.shape)
            #img = np.expand_dims(img, axis=0)
            #print("#"+str(img.shape))
            x_tmp_list.append(img)
            y_tmp_list.append(image_label)    
            n = n + 1
            limit_count = limit_count +1
            
            end_time = datetime.datetime.now()
            total_time = (end_time-start_time).total_seconds()
            all_total_time = all_total_time+total_time
            print("Load images progress : "+str(n-all_n_subtract)+" / "+str(batch_size) +" from " +str(all_n)+" , Remaining Time : "
                  +str(datetime.timedelta(seconds=total_time*(all_n-(n-all_n_subtract)))).split('.')[0]
                  +" , Total Time : "+str(datetime.timedelta(seconds=all_total_time)).split('.')[0], end="\r")
            
    print("Load images complete \n")
    return np.array(x_tmp_list),np.array(y_tmp_list),n