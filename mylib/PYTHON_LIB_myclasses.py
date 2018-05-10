import matplotlib.pyplot as plt
import numpy as np
import keras
from IPython.display import clear_output

class ImageObject:
        def __init__(self,rootpath,weather,timestring,category,filename):
                self.rootpath = rootpath
                self.weather = weather
                self.timestring = timestring
                self.category = category
                self.filename = filename
                
        def GetTrueAnswer(self):
            if (self.category == "Occupied"):
                return True
            else:
                return False
            
        def GetImageFilePath(self):
            return self.rootpath+"/"+self.weather+"/"+self.timestring+"/"+self.category+"/"+self.filename

class FileWriterObject:
    
    file_dir = ""
    
    def __init__(self,f_name):
        self.file_dir = f_name
        
    def SetFileDirectory(self,f_name):
        self.file_dir = f_name
    
    def Write(self,text):
        file = open(self.file_dir,"a")
        file.write(text+"\n") 
        file.close() 
        
class ConfusionMatrixObject:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0
        
    def AddValueToConfusionMat(self,val,label_val):
        if (val == 1 and val == label_val):
            self.true_positive = self.true_positive + 1
        elif (val == 1 and val != label_val):
            self.false_positive = self.false_positive + 1
        elif (val == 0 and val != label_val):
            self.false_negative = self.false_negative + 1
        else:
            self.true_negative = self.true_negative + 1
    
    def Accuracy(self):
        return (self.true_positive + self.true_negative)/(self.true_positive+self.false_positive+self.false_negative+self.true_negative)
    
    def Precision(self):
        return (self.true_positive)/(self.true_positive+self.false_positive)
    
    def Recall(self):
        return (self.true_positive)/(self.true_positive+self.false_negative)
    
    def F1Score(self):
        return (2*(self.Recall() *self.Precision()))/(self.Recall() +self.Precision())
    
    def PrintConfusionMat(self):
        return "TRUE POSITIVE = {"+str(self.true_positive)+"} || FALSE POSITIVE = {"+str(self.false_positive)+"} || FALSE NEGATIVE = {"+str(self.false_negative)+"} || TRUE NEGATIVE = {"+str(self.true_negative)+"}\n"
    
    def PrintAccuracy(self):
        val = self.Accuracy()
        return "Accuracy = "+str(val)+" ( "+str(val*100)+" % )\n"
    
    def PrintPrecision(self):
        val = self.Precision()
        return "Precision = "+str(val)+" ( "+str(val*100)+" % )\n"
    
    def PrintRecall(self):
        val = self.Recall()
        return "Recall = "+str(val)+" ( "+str(val*100)+" % )\n"
    
    def PrintF1Score(self):
        val = self.F1Score()
        return "F1 Score = "+str(val)+" ( "+str(val*100)+" % )\n"
    
class CaffeFunctionUserDefine:
    def vis_square(self,data):
        #"""Take an array of shape (n, height, width) or (n, height, width, 3)
        #   and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                   (0, 1), (0, 1))                 # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        plt.imshow(data); plt.axis('off')
        
class ImageCropPositionObject:
    start_x = 0
    start_y = 0
    width = 0
    height = 0
    
    def __init__(self,x,y,w,h):
        self.start_x = x
        self.start_y = y
        self.width = w
        self.height = h
        
    def GetDestinationPosition(self,pos):
        if(pos=="x"):
            return self.start_x + self.width
        elif(pos=="y"):
            return self.start_y + self.height

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
        
class ImageCropSet:
    start_x = 0
    start_y = 0
    width = 0
    height = 0
    
    def __init__(self,x,y,w,h):
        self.start_x = x
        self.start_y = y
        self.width = w
        self.height = h
        
    def GetDestinationPosition(self,pos):
        if(pos=="x"):
            return self.start_x + self.width
        else:
            return self.start_y + self.height

class TimeCounter:
    start_time = 0
    end_time = 0
    start_count_flag = False
    name = ""
    
    def __init__(self,name):
        self.name = name
        self.mini_time_counter = []

    def StartCounting(self):
        self.start_count_flag = True
        self.start_time = time.time()

    def StopCounting(self):
        if (self.start_count_flag == True):
            self.start_count_flag = False
            self.end_time = time.time()

    def GetElapsedTime(self):
         if (self.start_count_flag == False):
             return self.end_time - self.start_time

    def PrintElapsedTime(self):
        elapsed_time = self.GetElapsedTime()
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    def CreateMiniCounter(self):
        idx = len(self.mini_time_counter)
        self.mini_time_counter.append(TimeCounter(self.name+"_mini"+str(idx)))
        return idx
    
    def GetMiniTimeCounter(self,idx):
        return self.mini_time_counter[idx]
    
def FirebaseSendData(predict_set):
   ## Add data to firebase
    epoch_time_now = calendar.timegm(time.gmtime())
    for i ,val in predict_set.items():
        conv_val = np.asscalar(val)
        respond = firebase.put("/car-park-snapshots/swu/park0/p"+str(i), "val",conv_val )
        history_respond = firebase.put("/car-park-snapshots-history/swu/park0/p"+str(i)+"/"+str(epoch_time_now),"val", conv_val)
