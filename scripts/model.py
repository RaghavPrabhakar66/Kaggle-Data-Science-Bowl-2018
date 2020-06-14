import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import random
import sys

import tensorflow as tf
from tensorflow import keras

# BUILDING MODEL 

def down_conv_block(x, filters, kernel_size=(3, 3), padding='SAME', strides=1):
    c = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    c = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(c)
    p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c)

    return c, p

def up_conv_block(x, skip, filters, kernel_size=(3, 3), padding='SAME', strides=1):
    us     = keras.layers.UpSampling2D(size=(2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c      = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(concat)
    c      = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding='SAME', strides=1):
    c = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    c = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(c)    

    return c

def UNET():
    input = keras.layers.Input((128, 128, 3))

    c1, p1 = down_conv_block(x=input, filters=16)
    c2, p2 = down_conv_block(x=p1, filters=32)
    c3, p3 = down_conv_block(x=p2, filters=64)
    c4, p4 = down_conv_block(x=p3, filters=128)

    bn = bottleneck(x=p4, filters=256)

    u1 = up_conv_block(x=bn, skip=c4, filters=128)
    u2 = up_conv_block(x=u1, skip=c3, filters=64)
    u3 = up_conv_block(x=u2, skip=c2, filters=32)
    u4 = up_conv_block(x=u3, skip=c1, filters=16)

    outputs = keras.layers.Conv2D(filters=1, kernel_size=(1,1), padding='SAME', activation='sigmoid')(u4)

    model = keras.models.Model(input, outputs)

    return model

