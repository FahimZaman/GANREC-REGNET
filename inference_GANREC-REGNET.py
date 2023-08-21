#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fahim Ahmed Zaman
"""

#%% Necessary Libraries
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utilities import models, tools

#%% Load data
image_path = './data/demoData/demoImage.nii.gz'
label_path = './data/demoData/demoLabel.nii.gz'

image, label = tools.read_image(image_path, label_path)
# Extract ROI
imageROI, labelROI = tools.select_ROI(image, label)
# Binary Label
labelROI[labelROI>0] = 1

I_DEPTH, I_HEIGHT, I_WIDTH = imageROI.shape

#%% Define Models
tf.keras.backend.clear_session()
image_shape_RECNET = (I_DEPTH, I_HEIGHT, I_WIDTH, 1)
image_shape_REGNET = (I_DEPTH, I_HEIGHT, I_WIDTH, 2)
RECNET = models.define_RECNET(image_shape_RECNET)
REGNET = models.define_REGNET(image_shape_REGNET)

#%% Load Model Weights
RECNET_weights_path = './modelWeights/KneeRECNET.hdf5'
REGNET_weights_path = './modelWeights/KneeREGNET.hdf5'
RECNET.load_weights(RECNET_weights_path)
REGNET.load_weights(REGNET_weights_path)

#%% Reconstruct Image
imageConditionedOnLabels = np.where(labelROI==1,0,imageROI)
imageReconstructed = np.squeeze(RECNET.predict(np.expand_dims(imageConditionedOnLabels, axis=(0,-1)), batch_size=1))
# Difference Image
imageDifference = np.abs(imageROI-imageReconstructed)
    
#%% Predict Patch-wise DSC
# Stack Difference & Label Image
imageStacked = np.expand_dims(np.stack((imageDifference, labelROI), axis=-1), axis=0)
patchwiseDSC = np.squeeze(REGNET.predict(imageStacked, batch_size=1))

# Voxel-wise DSC scores from patch
DSC = tools.interpolate_DSC(patchwiseDSC, imageROI)

#%%
# Edge image from label
labelEdges = tools.surface_edge(labelROI)

# Interactive plot
fig, ax = plt.subplots()
imagePlot = tools.image_color(imageROI[0], labelEdges[0])
ax.imshow(imagePlot, cmap='gray')
ax.imshow(DSC[0], cmap='jet_r', alpha=0.2)
ax.axis('off')
ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
slider = Slider(ax_slider, 'Slide->', 1, len(imageROI)+1, valinit=1)
def update(value):
    nslice = int(value)-1
    imagePlot = tools.image_color(imageROI[nslice], labelEdges[nslice])
    ax.imshow(imagePlot, cmap='gray')
    ax.imshow(DSC[nslice], cmap='jet_r', alpha=0.2)
    ax.axis('off')
    slider.valtext.set_text('SliceNo-{}'.format(int(value)))
    fig.canvas.draw_idle()
slider.on_changed(update)
plt.show()