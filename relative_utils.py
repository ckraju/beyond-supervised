#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import cv2
import torch
from torch.utils import data
from random import shuffle
import os.path
import os
import copy
import math

class RelativeTileDataLoader(data.Dataset):

    def __init__(self, img_root, image_list, crop_shape, mirror = True, split = 'train'):
        self.img_root = img_root
        self.split = split
        self.image_list = [line.rstrip('\n') for line in open(image_list)]

        self.mirror = mirror
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.std_bgr = 255*np.array([0.229, 0.224, 0.225])
        self.crop_shape = crop_shape


        self.files = collections.defaultdict(list)
        for f in self.image_list:
            self.files[self.split].append({'img': img_root+f, 'lbl': 0})
        
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        """ get the image"""
        image_file_name = self.img_root + self.image_list[index]
        
        image = None
        if os.path.isfile(image_file_name):
            image = cv2.imread(image_file_name)
        else:
            print('ERROR: couldn\'t find image -> ', image_file_name)
            
        if self.mirror:
            flip = torch.LongTensor(1).random_(0, 2)[0]*2-1
            image = image[:, ::flip, :]

        
        """ Divide image into 3x3"""
        tile_shape = (image.shape[0]+2)/3, (image.shape[1]+2)/3 # add 2 to consider non multiples of 3
        center_tile_topleft_corner= tile_shape
        
        """ get random crop location from center tile """
        center_crop_topleft_corner = (int(np.random.randint(0,tile_shape[0] - self.crop_shape[0]-1) + center_tile_topleft_corner[0]) \
                                   , int(np.random.randint(0,tile_shape[1] - self.crop_shape[1]-1) + center_tile_topleft_corner[1]))
        
        """ choose random tile location out of the 8 neighbouring """
        possible_tile_locs = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
#         chosen_tile_idx = np.random.randint(0,len(possible_tile_locs))
        chosen_tile_idx = torch.LongTensor(1).random_(len(possible_tile_locs))[0]
        chosen_tile_loc = possible_tile_locs[chosen_tile_idx]
        
        """ Get a random crop location out of this tile """
        random_tile_topleft_corner = (chosen_tile_loc[0] * tile_shape[0]) , (chosen_tile_loc[1] * tile_shape[1])
        random_crop_topleft_corner = (int(np.random.randint(0,tile_shape[0] - self.crop_shape[0]-1) + random_tile_topleft_corner[0]) \
                                      , int(np.random.randint(0,tile_shape[1] - self.crop_shape[1]-1) + random_tile_topleft_corner[1]) )
        

        
        """ Get the actual crops """
        center_crop = image[center_crop_topleft_corner[0]:center_crop_topleft_corner[0]+self.crop_shape[0] , \
                            center_crop_topleft_corner[1]:center_crop_topleft_corner[1]+self.crop_shape[1], \
                            :]

        random_crop = image[random_crop_topleft_corner[0]:random_crop_topleft_corner[0]+self.crop_shape[0] , \
                            random_crop_topleft_corner[1]:random_crop_topleft_corner[1]+self.crop_shape[1], \
                            :]


        return self.transform_image(center_crop),self.transform_image(random_crop),chosen_tile_idx,chosen_tile_loc #torch.from_numpy(chosen_tile_idx),chosen_tile_loc


    def transform_image(self, image):
        image = image.astype(np.float64)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()

        return image


