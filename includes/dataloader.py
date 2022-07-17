import cv2
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
from glob import glob

from .dataset import GIImageDataset
from sklearn.model_selection import train_test_split
import numpy as np



def get_loader(
		batchSize = 32,
		device = "cuda",
		numWorkers = 2,
		shuffle = False,
		dataset = None,
		transforms = None,
		shape = (288, 288)
	):
    
	if(dataset is None):
		return None

	dataset = GIImageDataset(dataset, shape, transforms)
	return DataLoader(dataset, batch_size = batchSize, num_workers = numWorkers, shuffle = shuffle)



def read_csv(csvFile, basePath = ""):
	base_path = basePath

	train_df = pd.read_csv(os.path.join(base_path, csvFile))

	train_df = pd.pivot_table(train_df, values='segmentation', index='id',columns='class',aggfunc=np.max).astype(str).fillna('')
	train_df = train_df.reset_index()

	x = glob(base_path + 'train/*/*/*/*')

	#storing image path as the id and all the corresponding info
	def setup_img_info(x):
	    name = x.split('/')[-1].split('_')
	    img_name = ""
	    img_name += x.split('/')[-3] + '_' + name[0]+'_'+name[1]
	    height,width,h_pixel,w_pixel = (name[2],name[3],name[4],name[5].replace('.png',''))
	    #print(img_name , height,width,h_pixel,w_pixel)
	    return pd.Series([img_name , height ,width ,h_pixel ,w_pixel])
	image_detail = pd.DataFrame()
	image_detail['img_path'] = x
	image_detail[['id','height','width','h_pixel','w_pixel']] = image_detail['img_path'].apply(lambda t : setup_img_info(t))
	    
	train_df = pd.merge(image_detail , train_df , on='id')
	
	df_train, df_valid = train_test_split(train_df.values, test_size=0.2, shuffle = True)

	return df_train, df_valid


def prepare_loaders(batchSize = 32,
		device = "cuda",
		numWorkers = 2,
		shuffle = False,
		csvFile = None,
		basePath = None,
		shape = (288, 288)
	):
	
	train_data, valid_data = read_csv(csvFile, basePath)

	train_transform = A.Compose(
		[
		    A.HorizontalFlip(p=0.5),
		    A.VerticalFlip(p=0.1),
# 			A.OneOf([
# 		        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
# 		        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.50),
# 		    ], p=0.5),
		    ToTensorV2()#transpose_mask = True),
		]
	)
	valid_transform = A.Compose(
		[
		    ToTensorV2()#transpose_mask = True),
		],
	)

	train_loader = get_loader(batchSize, device, numWorkers = 2, shuffle = False, dataset = train_data, transforms = train_transform, shape = shape)
	val_loader = get_loader(batchSize, device, numWorkers = 2, shuffle = False, dataset = valid_data, transforms = valid_transform, shape = shape)
	
	return train_loader, val_loader



