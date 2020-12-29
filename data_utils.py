
#=============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import h5py
import torch
from PIL import Image, ImageOps
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from sklearn.cross_validation import StratifiedShuffleSplit
    # cross_validation -> now called: model_selection
    # https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
#=============================================================================


SHAPE = (288, 128, 128, 8)
 
class train_generator():

    def __init__(self, filenames, batch_size, training_size, height, width):
        self.filenames = filenames
        self.batch_size = batch_size
        self.training_size = training_size
        self._train_set = torch.zeros(SHAPE)
        self._height = height
        self._width = width
        self._train_cat_data()
        
    # Readin all the datasets and concatenate them    
    def _train_cat_data(self):
        for i in range(len(self.filenames)):
                print('Now we are processing the {}th file'.format(i))
                
                try:
                    filename = self.filenames[i]
                    f = h5py.File('data/training/'+filename, 'r+')
                except FileNotFoundError as fnf_error:
                    print(fnf_error)
                    
                if i == 0:
                    temp = f['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._train_set = torch.FloatTensor(temp)
                else:
                    temp = f['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._train_set = torch.cat((self._train_set, torch.FloatTensor(temp)), dim=0)
    
    def get_data(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        return self._train_set
    
    def get_cuda(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        if torch.cuda.is_available():
            self._train_set = self._train_set.cuda()
        else:
            print("GPU is not available on your computer.")
    
#     def get_train(self):       
#         data_for_train = torch.zeros(self.batch_size, (self.training_size -3)*8, self._height, self._width)
#         data_for_pred = torch.zeros(self.batch_size, 3*8, self._height, self._width)
#         i = 0
#         j = 0
#         while True:      
#             temp1 = self._train_set[j:(j + self.training_size-3), :,:,:] # Get part of the image for coding
#             temp1 = temp1.reshape(1, (self.training_size-3)*8, self._height, self._width) 
#             data_for_train[i,:,:,:] = temp1
#             temp2 = self._train_set[(j + self.training_size -3):(j + self.training_size),:,:,:]
#             temp2 = temp2.reshape(1, 3*8, self._height, self._width)
#             data_for_pred[i,:,:,:] = temp2
#             i += 1
#             j += 1
#             if i >= self.batch_size: # if i reaches the batch size, yield the dataset
#                 print("This is the {}th batch".format(j//self.batch_size))
#                 yield [data_for_train, data_for_pred]
#                 i = 0
#                 # If the remaining data cannot fill the 'data_for_train', we would not like to proceed again
#                 # Just break and stop the loop
#                 if j >= (self._train_set.shape[0] - self.training_size - self.batch_size):  
#                     print("All training data has been used")
#                     break
                    
    def get_train_new(self):       
        data_for_train = torch.zeros(self.batch_size, (self.training_size -3)*8, self._height, self._width)
        data_for_pred = torch.zeros(self.batch_size, 3*8, self._height, self._width)
        i = 0
        j = 0
        m = 0
        while True: 
            temp1 = self._train_set[j:(j + self.training_size-3), :,:,:] # Get part of the image for coding
            temp1 = temp1.reshape(1, (self.training_size-3)*8, self._height, self._width) 
            data_for_train[i,:,:,:] = temp1
            temp2 = self._train_set[(j + self.training_size -3):(j + self.training_size),:,:,:]
            temp2 = temp2.reshape(1, 3*8, self._height, self._width)
            data_for_pred[i,:,:,:] = temp2
            i += 1  
            j = 50 + j
            if i >= self.batch_size: # if i reaches the batch size, yield the dataset
                print("This is the {}th batch in training".format(m+1))
                yield [data_for_train, data_for_pred]
                i = 0
                m += 1
                j = m
                # If the remaining data cannot fill the 'data_for_train', we would not like to proceed again
                # Just break and stop the loop
                if j >= (self._train_set.shape[0] - self.training_size - self.batch_size):  
                    print("All training data has been used")
                    break
                    
                    
class valid_generator():

    def __init__(self, filenames, batch_size, validation_size, height, width):
        self.filenames = filenames
        self.batch_size = batch_size
        self._vali_set = torch.zeros(SHAPE)
        self._vali_size = validation_size
        self._height = height
        self._width = width
        self._validation_cat_data()
    
    # Readin all the datasets and concatenate them
    def _validation_cat_data(self):
        for i in range(len(self.filenames)):
                print('Now we are processing the {}th file'.format(i))
                try:
                    filename = self.filenames[i]
                    g = h5py.File('data/validation/'+filename, 'r+')
                except FileNotFoundError as fnf_error:
                    print(fnf_error)
                
                if i == 0:
                    temp = g['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._vali_set = torch.FloatTensor(temp)
                else:
                    temp = g['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._vali_set = torch.cat((self._vali_set, torch.FloatTensor(temp)), dim=0)
    
    def get_data(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        return self._vali_set
    
    def get_cuda(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        if torch.cuda.is_available():
            self._vali_set = self._vali_set.cuda()
        else:
            print("GPU is not available on your computer.")
    
                    
    def gen_valid(self):
        data_for_input = torch.zeros(self.batch_size, (self._vali_size -3)*8, self._height, self._width)
        data_for_pred = torch.zeros(self.batch_size, 3*8, self._height, self._width)
        i = 0
        j = 0
        m = 0
        while True: 
            temp1 = self._vali_set[j:(j + self._vali_size-3), :,:,:] # Get part of the image for coding
            # print(temp1.shape)
            temp1 = temp1.reshape(1, (self._vali_size-3)*8, self._height, self._width) 
            
            data_for_input[i,:,:,:] = temp1
            temp2 = self._vali_set[(j + self._vali_size -3):(j + self._vali_size),:,:,:]
            temp2 = temp2.reshape(1, 3*8, self._height, self._width)
            data_for_pred[i,:,:,:] = temp2
            i += 1  
            j += 20
            if i >= self.batch_size: # if i reaches the batch size, yield the dataset
                print("This is the {}th batch in validation".format(m+1))
                yield [data_for_input, data_for_pred]
                i = 0
                m += 1
                j = m
                # If the remaining data cannot fill the 'data_for_validation', we would not like to proceed again
                # Just break and stop the loop
                if j >= (self._vali_set.shape[0] - self._vali_size - self.batch_size):  
                    print("All validation data has been used")
                    m = 0 # Reuse the data
                break
    
class test_generator():  
    def __init__(self, filenames, height, width):
        self.filenames = filenames
        self._height = height
        self._width = width
        self._test_set = torch.zeros(SHAPE)
        self._test_cat_data()
        
    def _test_cat_data(self):
        for i in range(len(self.filenames)):
                print('Now we are processing the {}th file'.format(i))
                filename = self.filenames[i]
                #print(filename)
                g = h5py.File('data/test/'+filename, 'r+')
                if i == 0:
                    temp = g['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._test_set = torch.FloatTensor(temp)
                else:
                    temp = g['array']
                    temp = temp[:,275:403,150:278,0:8]
                    self._test_set = torch.cat((self._vali_set, torch.FloatTensor(temp)), dim=0)
                    
    def get_data(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        return self._test_set
    
    def first_12_frames(self): # Get the whole dataset, with original shape
    #    self._cat_data()
        return self._test_set[0:12,:,:,:].reshape(1,8,self._height, self._width)
        
    def plot(self, output):    
        # Here we only plot a few frames
        frames = [12,13,14]
        truth = self._test_set[12:15,:,:,:]
        for i in range(3):   
                predict_plot = torch.zeros(128,128,3,dtype = int)
                ground_truth = torch.zeros(128,128,3,dtype = int)
                
                #Read in the prediction
                #We just use red color
                assert torch.sum(output) != 0
                
                output1 = output.reshape(3, self._height, self._width, 8)
                #print('difference2',torch.sum(output1[0,:,:,:] - self._test_set[12,:,:,:]),i)
                predict_plot[:,:,0] = output1[i,:,:,0]
                print('predict_plot sum',torch.sum(predict_plot))
                
                #Read in the ground truth
                
                ground_truth[:,:,0] = truth[i,:,:,0]
              # print('difference',torch.sum(predict_plot - ground_truth),i)
            
                plt.figure(figsize = (20,10))
                plt.imshow(predict_plot)
                plt.axis('off')
                plt_path1 = 'pic_predict/predict%d.png'%i
                plt.savefig(plt_path1, dpi=300, bbox_inches='tight')
                
                plt.figure(figsize = (20,10))
                plt.imshow(ground_truth)
                plt.axis('off')
                plt_path2 = 'pic_truth/truth%d.png'%i
                plt.savefig(plt_path2, dpi=300, bbox_inches='tight')

