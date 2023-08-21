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
EPOCHS = 5
BATCH_SIZE = 2

#%% Data Load
# Data path
image_path = './data/demoTrain/images/'
label_path = './data/demoTrain/labels/'
# Data read
images, labels = tools.read_data(image_path, label_path)
# Conditioned image using labels
conditionedImages = np.where(labels>0, 0, images)

#%% Train-Validation split
X_train, X_val, y_train, y_val = train_test_split(conditionedImages, images, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
image_shape = X_train.shape[1:]

#%% Model Training
# Define the models
tf.keras.backend.clear_session()
discriminatorModel = models.define_discriminator(image_shape)
RECNET = models.define_RECNET(image_shape)
GANRECNET = models.define_GANRECNET(RECNET, discriminatorModel, image_shape)
# Train the models
real_loss, fake_loss, GAN_loss, val_loss = tools.train_GANRECNET(discriminatorModel, RECNET, GANRECNET,
                                                                 X_train, y_train, X_val, y_val,
                                                                 n_epochs = EPOCHS, n_batch = BATCH_SIZE)

#%% Plot Results
tf.keras.backend.clear_session()
tools.plot_trainLoss(real_loss, fake_loss, GAN_loss, val_loss)
tools.plot_checkRECNET(X_train, y_train)
_ = tools.predictions_RECNET(X_train, y_train)