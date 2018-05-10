import os
import numpy as np
import keras
import time

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
    
    def StartCounting(self):
        self.start_count_flag = True
        self.start_time = time.time()
        self.end_time = time.time()

    def StopCounting(self):
        if (self.start_count_flag == True):
            self.start_count_flag = False
            self.end_time = time.time()

    def GetElapsedTime(self):
         if (self.start_count_flag == False):
            return self.end_time - self.start_time

    def PrintElapsedTime(self):
        if (self.start_count_flag == False):
            elapsed_time = self.GetElapsedTime()
            return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
