#!/usr/bin/env python

import numpy as np
import cv2
import torch
from torch.utils import data
from random import shuffle
import os.path
import collections



class SegmentationDataLoader(data.Dataset):
    
    def __init__(self, img_root, gt_root, image_list, split = 'train',
                     mirror = True, mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434]),   
                     transform=True):
        
        np.random.seed(7)
        ### paths
        self.img_root = img_root
        self.gt_root = gt_root

        self.split = split
        
        ### list of all images
        self.image_list = [line.rstrip('\n') for line in open(image_list)]
        
        
        ### augmentations
        self.mirror = mirror
           
        
        self._transform = transform
        self.mean_bgr=mean_bgr
        
        self.files = collections.defaultdict(list)
        for f in self.image_list:
            self.files[self.split].append({'img': img_root+f, 'lbl': gt_root+f})
        
    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        
        image_file_name = self.img_root + self.image_list[index]
        seg_gt_name = self.gt_root + self.image_list[index]
        
        ### read image
        image = None
        if os.path.isfile(image_file_name):
            image = cv2.imread(image_file_name)
        else:
            print('ERROR: couldn\'t find image -> ', image_file_name)
        
            
        ### read segmentation gt as greyscale image
        seg_gt = None
        if os.path.isfile(seg_gt_name):
            gt = cv2.imread(seg_gt_name)
            seg_gt = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
            ind1 = np.where(gt[:,:,1] == 255)
            seg_gt[ind1] = 1
            ind2 = np.where(gt[:,:,2] == 255)
            seg_gt[ind2] = 2
        else:
            print('ERROR: couldn\'t find segmentation gt ->', seg_gt_name)
        
        
        ### apply mirroring
        if self.mirror:
            flip = torch.LongTensor(1).random_(0, 2)[0]*2-1
            image = image[:, ::flip, :]
            seg_gt = seg_gt[:, ::flip]

            
        
        ### shuffle list after epoch
        if index == len(self.image_list):
            shuffle(self.indexlist)
            
        if self._transform:
            return self.transform(image, seg_gt)
            
    

    def transform(self, img, lbl):
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        return img, lbl
