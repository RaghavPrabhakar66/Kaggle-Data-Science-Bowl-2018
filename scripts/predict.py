import os
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from DataGenerator import DataGen

def iou(y_true, y_pred):
     def f(y_true, y_pred):
         intersection = (y_true * y_pred).sum()
         union = y_true.sum() + y_pred.sum() - intersection
         x = (intersection + 1e-15) / (union + 1e-15)
         x = x.astype(np.float32)
         return x
     return tf.numpy_function(f, [y_true, y_pred], tf.float32)

with CustomObjectScope({'iou': iou}):
         model = load_model(r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\files\save_model\model.h5')
test_path  = r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\test'

def mask_parse(mask):
     mask = np.squeeze(mask)
     mask = [mask, mask, mask]
     mask = np.transpose(mask, (1, 2, 0))
     return mask


def predict(path, id):
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image/255.0

    y_pred = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    h, w, _ = image.shape

    white_line = np.ones((h, 1, 3)) * 200.0
    all_images = [image * 255.0, white_line, mask_parse(y_pred) * 255.0]
    image = np.concatenate(all_images, axis=1)
    filename1 = os.path.join(r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\test\final', id)
    filename2 = os.path.join(r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\test\masked', id)
    cv2.imwrite(filename1, image)
    cv2.imwrite(filename2, mask_parse(y_pred) * 255.0)


test_ids = os.listdir((os.path.join(test_path, "images")))

for id in test_ids:
    path = os.path.join(test_path, "images", id)
    print(path )
    predict(path, id)