import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import random
import sys

import tensorflow as tf
from tensorflow import keras

class DataGen(keras.utils.Sequence):
    def __init__(self, path, batch_size=8, img_size=128, shuffle=True):
        self.path       = path
        self.id         = os.listdir((os.path.join(self.path, "images")))
        self.batch_size = batch_size
        self.img_size   = img_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __load__(self, id_name):
        image_path  = os.path.join(self.path, "images", id_name)
        masked_path = os.path.join(self.path, "masked", id_name)

        
        image = cv2.imread(image_path, 1)   # Reading Image in RGB format
        image = cv2.resize(image, (self.img_size, self.img_size))

        mask_image = cv2.imread(masked_path, -1)    # Reading image in grayscale format
        mask_image = cv2.resize(mask_image, (self.img_size, self.img_size))   # Image_Size = 128 X 128

        # Normalizing the image
        image = image/255.0
        mask_image  = mask_image/255.0

        return image, mask_image

    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.id):
            file_batch = self.id[index*self.batch_size:]
        else:
            file_batch = self.id[index*self.batch_size:(index+1)*self.batch_size]
        
        images, masks = [], []

        for id_name in file_batch:
            _img, _mask = self.__load__(id_name)
            images.append(_img)
            masks.append(_mask)

        images = np.array(images)
        masks  = np.array(masks)

        return images, masks
    
    
    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.id)/float(self.batch_size)))
