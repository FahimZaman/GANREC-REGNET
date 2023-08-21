#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fahim Ahmed Zaman
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from utilities import tools, models

#%% Set Train Parameters
EPOCHS = 10
BATCH_SIZE = 2

#%% Data Load
# Data path
image_path = './data/demoTrain/images/'
label_path = './data/demoTrain/labels/'
segmentation_path = './data/demoTrain/segmentations/'
# Data read
images, labels, segmentations = tools.read_data(image_path, label_path, segmentation_path)
# Conditioned image using labels
conditionedImages = np.where(labels>0, 0, images)
# RECNET reconstructed image
reconstructedImages = tools.predictions_RECNET(np.expand_dims(conditionedImages, axis=-1))
# Difference image
imageDifference = np.abs(images-reconstructedImages)
# Stack Difference & Label Image
imageStacked = np.stack((imageDifference, segmentations), axis=-1)

#%% Model training
# DSC calculation
patchwiseDSC = tools.patchwise_DSC(labels, segmentations)
# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(imageStacked, patchwiseDSC, test_size=0.2, shuffle = False, stratify = None)
image_shape = X_train.shape[1:]
# Define REGNET
tf.keras.backend.clear_session()
REGNET = models.define_REGNET(image_shape)
# Train the model
tools.train_REGNET(REGNET, 
                   X_train, y_train,
                   X_val, y_val,
                   n_epochs = EPOCHS,
                   n_batch = BATCH_SIZE)