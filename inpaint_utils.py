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



class ContextInpaintingDataLoader(data.Dataset):
	
	def __init__(self, img_root, image_list, mirror = True, 
		context_shape = [32, 32], context_count = 4, split = 'train'):

		self.img_root = img_root
		self.split = split
		self.image_list = [line.rstrip('\n') for line in open(image_list)]

		self.mirror = mirror
		self.context_shape = context_shape
		self.context_count = context_count

		self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.std_bgr = 255*np.array([0.229, 0.224, 0.225])


		self.files = collections.defaultdict(list)
		for f in self.image_list:
			self.files[self.split].append({'img': img_root+f, 'lbl': 0})
		
	def __len__(self):
		return len(self.files[self.split])

	def __getitem__(self, index):
		
		image_file_name = self.img_root + self.image_list[index]
		
		image = None
		if os.path.isfile(image_file_name):
			image = cv2.imread(image_file_name)
		else:
			print('ERROR: couldn\'t find image -> ', image_file_name)
			
		if self.mirror:
			flip = torch.LongTensor(1).random_(0, 2)[0]*2-1
			image = image[:, ::flip, :]


		context_shape = self.context_shape

		context_mask = np.zeros((3, image.shape[0], image.shape[1]), np.uint8)

		if self.context_count == 1:
			left = image.shape[0]/2 - context_shape[0]/2
			context_mask[:, left:left+context_shape[0], left:left+context_shape[1]] = 1
			image[left:left+context_shape[0], left:left+context_shape[1], 0] = self.mean_bgr[0]
			image[left:left+context_shape[0], left:left+context_shape[1], 1] = self.mean_bgr[1]
			image[left:left+context_shape[0], left:left+context_shape[1], 2] = self.mean_bgr[2]

		else:
			orig_image = copy.deepcopy(image)
			for c_ in range(self.context_count):
				row = torch.LongTensor(1).random_(0, image.shape[0]-context_shape[0]-1)[0]
				col = torch.LongTensor(1).random_(0, image.shape[1]-context_shape[1]-1)[0]
				context_mask[:, row:row+context_shape[0], col:col+context_shape[1]] = 1
				image[row:row+context_shape[0], col:col+context_shape[1], 0] = self.mean_bgr[0]
				image[row:row+context_shape[0], col:col+context_shape[1], 1] = self.mean_bgr[1]
				image[row:row+context_shape[0], col:col+context_shape[1], 2] = self.mean_bgr[2]


		return self.transform_image(image), self.transform_mask(context_mask), self.transform_context(orig_image)

	def transform_image(self, image):
		image = image.astype(np.float64)
		image -= self.mean_bgr
		image = image.transpose(2, 0, 1)
		image = torch.from_numpy(image.copy()).float()

		return image

	def transform_mask(self, mask):
		
		mask = torch.from_numpy(mask.copy()).float()

		return mask

	def transform_context(self, context):
		context = context.astype(np.float64)
		context -= self.mean_bgr
		context[:,:,0] /= 3*self.std_bgr[0]
		context[:,:,1] /= 3*self.std_bgr[1]
		context[:,:,2] /= 3*self.std_bgr[2]

		index_ = context>1
		context[index_] = 1
		index_ = context<-1
		context[index_] = -1

		context = context.transpose(2, 0, 1)
		context = torch.from_numpy(context.copy()).float()

		return context