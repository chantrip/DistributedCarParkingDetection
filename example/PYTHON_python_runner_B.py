#import Necessary Libraries
import os
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import matplotlib.pyplot as plt
from mylib.PYTHON_LIB_myclasses import FileWriterObject , ConfusionMatrixObject

#Initialize constaint variables (DON'T CHANGE IT !!)
LINUX_UPPER_PATH = os.getcwd()
print("---Current Directory : " + LINUX_UPPER_PATH)
FOLDER_NAME = os.path.basename(LINUX_UPPER_PATH)
print("---Directory Name : " + FOLDER_NAME)

#Initialize Variables
batch_size = 1000
iteration_count = 999
image_color_channel = 3
image_size_w=224
image_size_h=224

#Initialize Files Directory
prototxt_path = LINUX_UPPER_PATH+"/Resources/NewModels/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt"
caffemodel_path = LINUX_UPPER_PATH+"/Resources/NewModels/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_870.caffemodel"
labels_path = LINUX_UPPER_PATH+"/Resources/SWUPark/test.txt"
means_images_npy_path = ""
root_images_folder = LINUX_UPPER_PATH+"/Resources/SWUPark/All"
text_output_filename = "TEXT_result_B_(mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val)_SWUPark-TEST_caffe"

#Initialize Necessary  Object
confusion_mat = ConfusionMatrixObject()
file_writer = FileWriterObject(f_name=LINUX_UPPER_PATH+"/"+text_output_filename+".final")
                       
print("---All Of Nesessary Initialize Complete---")

######################################################################################################
######################################################################################################
######################################################################################################

#Import Model
net = caffe.Net(prototxt_path,caffemodel_path,caffe.TEST)

#Set Preprocessor
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#(Swap) H x W x C >>> C x H x W
transformer.set_transpose('data', (2,0,1))
#(Swap) R : G : B >>> B : G : R
transformer.set_channel_swap('data', (2,1,0))
#(Set) For Color Depth 0-255 
#transformer.set_raw_scale('data', 255)
if (means_images_npy_path == ""):
    transformer.set_mean('data', np.array([0, 0, 0]))
    print("---Images Means is Set to Zero---")
    print("Mean-subtracted values:", zip("BGR", np.array([0, 0, 0])))
else:
    mu = np.load(means_images_npy_path).mean(1).mean(1)
    transformer.set_mean('data', mu) #Means From .npy Files
    print("---Images Means is Set by .npy Files---")
    print("Mean-subtracted values:", zip("BGR",mu))
    
net.blobs['data'].reshape(batch_size,image_color_channel,image_size_h, image_size_w)

######################################################################################################
######################################################################################################
###################################################################################################

file_writer.Write("====================================================================")
print("====================================================================")
print("for each layer, show the output shape")
file_writer.Write("for each layer, show the output shape")
for layer_name, blob in net.blobs.items():
    print(layer_name + "\t" + str(blob.data.shape))
    file_writer.Write(layer_name + "\t" + str(blob.data.shape))

print("====================================================================")
file_writer.Write("====================================================================")
print("The param shapes typically have the form (output_channels, input_channels, filter_height, filter_width) (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).")
file_writer.Write("The param shapes typically have the form (output_channels, input_channels, filter_height, filter_width) (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).")
for layer_name, param in net.params.items():
    print (layer_name + "\t" + str(param[0].data.shape), str(param[1].data.shape))
    file_writer.Write(layer_name + "\t" + str(param[0].data.shape)+" "+ str(param[1].data.shape))
print("====================================================================")
file_writer.Write("====================================================================")

######################################################################################################
######################################################################################################
###################################################################################################

#Declare Local Counter
line_read_count = 0
tmp_bsize = 0
tmp_iter_count=0
run_lists = []
first_time_in_each_iteration = True
all_of_line_count = 0

#Open Labels File And Read Lines 
with open(labels_path) as f:
    for x in f:  
        all_of_line_count = all_of_line_count+1
        
with open(labels_path) as f:
    for line in f:
        if (line_read_count < (1+iteration_count) * batch_size):
            #Check iter is end ?
            if(tmp_iter_count >= iteration_count):
                break;

            if (first_time_in_each_iteration):
                sub_list = []
                first_time_in_each_iteration = False
                
            #Get Image Directory String From The File
            txt_val = (root_images_folder+"/"+line.rstrip('\n')).split()

            #Declare Variables
            image_dir = txt_val[0]
            image_label = int(txt_val[1])

            prop_dict = {}
            prop_dict['img_path'] = image_dir
            prop_dict['truth_label'] = image_label
            prop_dict['batch_id'] = tmp_bsize
            prop_dict['iteration_id'] = tmp_iter_count
            prop_dict['real_line_pos'] = line_read_count
            
            sub_list.append(prop_dict)
            
            #Add Local Count
            tmp_bsize = tmp_bsize+1
            line_read_count = line_read_count+1
            
            #Reset tmp_bsize and run next iterate
            if(tmp_bsize >= batch_size):
                first_time_in_each_iteration = True
                tmp_bsize = 0
                tmp_iter_count = tmp_iter_count+1
                run_lists.append(sub_list)
            
            if (line_read_count == all_of_line_count):
                first_time_in_each_iteration = True
                tmp_bsize = 0
                tmp_iter_count = tmp_iter_count+1
                run_lists.append(sub_list)
                
        else:
            #Skip label line
            continue          
            
for i,sub_list in enumerate(run_lists):
    for j_0,prop_dict in enumerate(sub_list):
        #Load Image
        caffe_img = caffe.io.load_image(prop_dict['img_path'])
        
        #Preprocessing Image
        if (len(sub_list) != batch_size):
            net.blobs['data'].reshape(len(sub_list),image_color_channel,image_size_h, image_size_w)
        else:
            net.blobs['data'].reshape(batch_size,image_color_channel,image_size_h, image_size_w)
            
        net.blobs['data'].data[j_0,:,:,:] = transformer.preprocess('data', caffe_img)
    
    #Run Model
    out = net.forward()
    
    for j_1 , raw_predict_val in enumerate(out['score']):
        predict_val = raw_predict_val.argmax()
        
        #Storage Value
        prop_dict = sub_list[j_1]
        confusion_mat.AddValueToConfusionMat(val=predict_val,label_val=prop_dict['truth_label'])
        file_writer.Write("[["+str(prop_dict['real_line_pos'])+"]] "+" ["+str(prop_dict['iteration_id'])+"]"+" ["+str(prop_dict['batch_id'])+"] "+prop_dict['img_path']+" "+str(prop_dict['truth_label'])+" {"+str(predict_val)+"}")

    #Show confusion matrix every X iteration
    if (i < iteration_count):
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
    
print("====================================================================")
file_writer.Write("====================================================================")
print("---Test Completed---")
file_writer.Write("---Test Completed---")
print("====================================================================")
file_writer.Write("====================================================================")
print("---Summary---")
file_writer.Write("---Summary---")
print("batch size :"+str(batch_size))
file_writer.Write("batch size : "+str(batch_size))
print("all iteration count : "+str(len(run_lists)))
file_writer.Write("all iteration count : "+str(len(run_lists)))
print("TESTED IMAGES COUNT : "+str(line_read_count))
file_writer.Write("TESTED IMAGES COUNT : "+str(line_read_count))
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
        
        