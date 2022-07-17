import cv2
import numpy as np
from torch.utils.data import Dataset
from .utils.preprocess_fn import padImage, rle_decode

class GIImageDataset(Dataset):
    
    def __init__(self, image_dataframe, shape, transforms = None):
        super(GIImageDataset, self).__init__()
        
        self.image_dataframe = image_dataframe
        self.transforms = transforms
        self.IMG_W = shape[0]
        self.IMG_H = shape[1]
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, index):
        #extract file name
        path = self.image_dataframe[index][0]
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        image = (image - image.min()) / (image.max() - image.min()) * 255.0  # scale image to [0, 255]
        image = padImage(image, (self.IMG_W, self.IMG_H))[0]
        
        mask = np.zeros((self.IMG_W, self.IMG_H, 3), dtype=np.float32)
        mask[:,:,0] = padImage(rle_decode(self.image_dataframe[index][6], shape=(int(self.image_dataframe[index][2]),int(self.image_dataframe[index][3]))), (self.IMG_W, self.IMG_H))[0]
        mask[:,:,1] = padImage(rle_decode(self.image_dataframe[index][7], shape=(int(self.image_dataframe[index][2]),int(self.image_dataframe[index][3]))), (self.IMG_W, self.IMG_H))[0]
        mask[:,:,2] = padImage(rle_decode(self.image_dataframe[index][8], shape=(int(self.image_dataframe[index][2]),int(self.image_dataframe[index][3]))), (self.IMG_W, self.IMG_H))[0]
        
        mask = np.transpose(mask, (2, 0, 1))
        for transform in self.transforms:

            augment = transform(image = image, mask = mask)

            image = augment["image"]
            mask = augment["mask"]
         
        return image, mask
