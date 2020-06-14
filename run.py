import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import random
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from scripts import model, DataGenerator

def iou(y_true, y_pred):
     def f(y_true, y_pred):
         intersection = (y_true * y_pred).sum()
         union = y_true.sum() + y_pred.sum() - intersection
         x = (intersection + 1e-15) / (union + 1e-15)
         x = x.astype(np.float32)
         return x
     return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# HYPERPARAMETERS

image_size =  128
train_path = r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\train'
val_path   = r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\validation'
test_path  = r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\test'
epochs     = 35
batch_size = 32
opt = tf.keras.optimizers.Adam(1e-4)
metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]

# COMPILING MODEL

model = model.UNET()
model.compile(loss="binary_crossentropy", 
              optimizer=opt, 
              metrics=metrics)


model.summary()

# TRAINING THE MODEL

train_datagen = DataGenerator.DataGen(path=train_path, batch_size=batch_size, img_size=image_size, shuffle=True)
valid_datagen = DataGenerator.DataGen(path=val_path, batch_size=batch_size, img_size=image_size, shuffle=False)

train_steps = len(os.listdir((os.path.join(train_path, "images")))) // batch_size
valid_steps = len(os.listdir((os.path.join(val_path, "images")))) // batch_size

callbacks = [ModelCheckpoint(r"C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\files\save_model\model.h5"),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
             CSVLogger(r"C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\files\data.csv"),
             TensorBoard(),
             EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)]

history = model.fit_generator(generator=train_datagen,
                              validation_data=valid_datagen,
                              steps_per_epoch=train_steps, 
                              validation_steps=valid_steps, 
                              epochs=epochs,
                              callbacks=callbacks)

print(" ")
print('\nhistory dict:', history.history)
print(" ")

test_ids = os.listdir((os.path.join(test_path, "images")))