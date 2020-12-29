#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:11:30 2020

@author: adam
"""

from time import sleep
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#new_length=150
#new_width=225
#startingpoint_x=275
#startingpoint_y=100

class RGBMapPlotter:
    def __init__(self, data,num_of_time,filename,type_of_data):
        self.num_of_time = num_of_time
        self.data = torch.round(data).detach().numpy()
        self.filename=filename
        self.type_of_data=type_of_data
    def plot_map(self):
        
        for j,data_slice in enumerate(self.data[:self.num_of_time,:,:,:]):
            
#             map_data,head = map_to_RGB_3(data_slice,128,128)
#             fig, axs = plt.subplots(2, 2,figsize=(15,15))
#             axs[0, 0].imshow(map_data[0,:,:])
#             axs[0, 0].set_title('NE')
#             axs[0, 0].axis('off')
#             axs[0, 1].imshow(map_data[1,:,:])
#             axs[0, 1].set_title('NW')
#             axs[0, 1].axis('off')
#             axs[1, 0].imshow(map_data[2,:,:])
#             axs[1, 0].set_title('SE')
#             axs[1, 0].axis('off')
#             axs[1, 1].imshow(map_data[3,:,:])
#             axs[1, 1].set_title('SW')
#             axs[1, 1].axis('off')
#             plt.axis('off')
#             plt_path = 'project test/pic/%d.png'%j
#             plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            
            map_data,head = map_to_RGB_2(data_slice,128,128)
            fig, axs = plt.subplots(2, 2,figsize=(15,15))
            axs[0, 0].imshow(map_data[0,:,:,:])
            axs[0, 0].set_title('NE Heading',fontsize=20)
            axs[0, 0].axis('off')
            axs[0, 1].imshow(map_data[1,:,:,:])
            axs[0, 1].set_title('NW Heading',fontsize=20)
            axs[0, 1].axis('off')
            axs[1, 0].imshow(map_data[2,:,:,:])
            axs[1, 0].set_title('SE Heading',fontsize=20)
            axs[1, 0].axis('off')
            axs[1, 1].imshow(map_data[3,:,:,:])
            axs[1, 1].set_title('SW Heading',fontsize=20)
            axs[1, 1].axis('off')
            plt.axis('off')
            plt_path = self.type_of_data + '/%d.png'%j
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')

#             map_data,head = map_to_RGB(data_slice,128,128)
#             plt.figure(figsize = (20,10))
#             plt.imshow(map_data)
#             plt.axis('off')
#             if self.type_of_data=="true":
#                 plt_path = 'project test/pic/true/%s.png'%str((self.filename)*3+j)
#                 plt.savefig(plt_path, dpi=300, bbox_inches='tight')
#             else:
#                 plt_path = 'project test/pic/prediction/%s.png'%str((self.filename)*3+j)
#                 plt.savefig(plt_path, dpi=300, bbox_inches='tight')

    
# def load_data(filenames,num_of_file,num_of_time,startingpoint_x,startingpoint_y,new_length,new_width):
        
#         for i in range(num_of_file):
#             filename = filenames[i]
#             f = h5py.File('project test/'+filename, 'r+')
#             data = f['array']
#             #data=data[startingpoint_x:startingpoint_x+new_length,startingpoint_y:startingpoint_y+new_width,:]
#             for j in range(num_of_time):
#                 yield data[j,startingpoint_x:startingpoint_x+new_length,startingpoint_y:startingpoint_y+new_width,:]


def map_to_RGB_3(data,new_length,new_width):
    ## create tensors
    
#We need to reduce the space being taken
#     tensor_vol =  torch.zeros(495,436,4)
    volume_placeholder = np.zeros([new_length,new_width,4])
    
#     tensor_head = torch.zeros(495,436,4)
    heading_placeholder = np.zeros([new_length,new_width,4])
    
#     new_tensor = torch.zeros(495,436,3,dtype = int)
    RGB_placeholder = np.zeros([4,new_length,new_width],dtype = float)
    



    for i in range(4):

        #sum all the volumn
        RGB_placeholder[i,:,:] += data[:,:,2*i]

#         #sum all the speed
#         RGB_placeholder[i,:,:,1] +=  data[:,:,2*i+1]
        
#         #no blue color, instead, 4 images
#         RGB_placeholder[i,:,:,2] +=  0
    
    
    return RGB_placeholder.astype('int32') ,volume_placeholder


def map_to_RGB_2(data,new_length,new_width):
        
    ## create tensors
    
#We need to reduce the space being taken
#     tensor_vol =  torch.zeros(495,436,4)
    volume_placeholder = np.zeros([new_length,new_width,4])
    
#     tensor_head = torch.zeros(495,436,4)
    heading_placeholder = np.zeros([new_length,new_width,4])
    
#     new_tensor = torch.zeros(495,436,3,dtype = int)
    RGB_placeholder = np.zeros([4,new_length,new_width,3],dtype = float)
    

    print(data.shape)

    for i in range(4):

        #sum all the volumn
        RGB_placeholder[i,:,:,0] = data[:,:,2*i]

        #sum all the speed
        RGB_placeholder[i,:,:,1] = data[:,:,2*i+1]
        
        #no blue color, instead, 4 images
        RGB_placeholder[i,:,:,2] +=  0
    
    
    return RGB_placeholder.astype('int32') ,volume_placeholder




def map_to_RGB(data,new_length,new_width):
        
    ## create tensors
#   We need to reduce the space being taken
#   tensor_vol =  torch.zeros(495,436,4)
    volume_placeholder = np.zeros([new_length,new_width,4])
    
#     tensor_head = torch.zeros(495,436,4)
    heading_placeholder = np.zeros([new_length,new_width,4])
    
#     new_tensor = torch.zeros(495,436,3,dtype = int)
    RGB_placeholder = np.zeros([new_length,new_width,3],dtype = float)
    
    direction_lst =  [1,85,170,255]
    #startingpoint_x=275
    #startingpoint_y=100
    #data=data[startingpoint_x:startingpoint_x+new_length,startingpoint_y:startingpoint_y+new_width,:]
    #print(data.shape)
    ##NE,NW,SE,SW = (0,1,2,3)
    for i in range(4):

        vol = data[:,:,2*i].flatten()

        ## if volumn value is not zero,we return 1 else 0.
        vol = np.array([x if x == 0 else 1 for x in vol ])


        ## if volumn value is not zero, we get the direction based on volumn position
        heading = np.array([x if x == 0 else direction_lst[i] for x in vol ])

        ## assign values to tensors
        volume_placeholder[:,:,i] = vol.reshape(new_length,new_width)

        heading_placeholder[:,:,i] =heading.reshape(new_length,new_width)
        



    for i in range(4):

        #sum all the volumn
        RGB_placeholder[:,:,0] += data[:,:,2*i]

        #sum all the speed
        RGB_placeholder[:,:,1] +=  data[:,:,2*i+1]
        

    #take the heading where the volumn is bigger
    volumn = data[:,:,[0,2,4,6]]
    volumn = np.moveaxis(volumn,[2,0],[0,1])
    heading_placeholder = np.moveaxis(heading_placeholder,[2,0],[0,1])
    n = np.argmax(volumn,0)
    RGB_placeholder[:,:,2] =0#np.choose(n,heading_placeholder)
#     RGB_placeholder=RGB_placeholder/4
    
    #assign values to the tensor

    
    
    #average volumn
#     RGB_placeholder[:,:,0] = np.array([int(x/y) if y !=0 else 0 for x,y in zip(RGB_placeholder[:,:,0].flatten(),np.sum(volume_placeholder,axis =2).flatten())]).reshape(new_length,new_width)

#     #average speed
#     RGB_placeholder[:,:,1] = np.array([int(x/y) if y !=0 else 0 for x,y in zip(RGB_placeholder[:,:,1].flatten(),np.sum(volume_placeholder,axis =2).flatten())]).reshape(new_length,new_width)

    return RGB_placeholder.astype('int32') ,volume_placeholder