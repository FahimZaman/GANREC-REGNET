#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fahim Ahmed Zaman
"""
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import SimpleITK as sitk
import os
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from utilities import models

def standardize(image):
    '''
    This function standardize the input image

    Parameters
    ----------
    image : n-D numpy array

    
    '''
    image = image - np.min(image)
    image = image / np.max(image)
    return image

def read_image(image_path, label_path, segmentation_path=False):
    '''
    This function read and standardize image and label from the given paths

    Parameters
    ----------
    image_path : image location
    label_path : label location
    segmentation_path : segmentation location
    
    Returns
    -------
    image, label, segmentation : 3D-numpy arrays
    '''
    # read image
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype((np.float32))
    # Standardize image
    image = standardize(image)
    # read label
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype((np.uint8))
    label[label>2] = 0
    images = [image, label]
    # read segmentation
    if segmentation_path is not False:
        segmentation = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).astype((np.uint8))
        segmentation[segmentation>2] = 0
        images = [image, label, segmentation]
    return images
    
def select_ROI(image, label, segmentation=False):
    '''
    This function extract ROI from image using the segmentation label

    Parameters
    ----------
    image : 3D-numpy array
    label : 3D-numpy label array (0:background, 1:Femur-Cartilage, 2:Tibia-Cartilage)
    segmentation : 3D-numpy segmentation array (0:background, 1:Femur-Cartilage, 2:Tibia-Cartilage)
    
    Returns
    -------
    imageROI : 3D-numpy array
    labelROI : 3D-numpy label array (0:background, 1:Femur-Cartilage, 2:Tibia-Cartilage)
    segmentationROI : 3D-numpy segmentation array (0:background, 1:Femur-Cartilage, 2:Tibia-Cartilage)
    '''
    # ROI Parameters
    I_DEPTH, I_HEIGHT, I_WIDTH = image.shape
    ROI_Height, ROI_startWidth, ROI_endWidth = 104, 50, 290
    
    # Mid Slice Selection
    if segmentation is not False:
        label1, label2 = np.where(segmentation==1,1,0), np.where(segmentation==2,1,0)
    else:
        label1, label2 = np.where(label==1,1,0), np.where(label==2,1,0)
    label1 = np.cumsum(np.sum(np.flip(label1[:,:,int(I_WIDTH/2)]),axis=0))
    label2 = np.cumsum(np.sum(label2[:,:,int(I_WIDTH/2)],axis=0))
    if np.asarray(np.nonzero(label2)).size>0 and np.asarray(np.nonzero(label1)).size>0:
        midSlice = int((I_HEIGHT-np.nonzero(label1)[0][0]+np.nonzero(label2)[0][0])/2)
    else:
        raise Exception('Label error! Please check the label mask.') 
    # Extract and standardize ROI
    imageROI = image[:,midSlice-int(ROI_Height/2):midSlice+int(ROI_Height/2),ROI_startWidth:ROI_endWidth]
    labelROI = label[:,midSlice-int(ROI_Height/2):midSlice+int(ROI_Height/2),ROI_startWidth:ROI_endWidth]
    imageROI = standardize(imageROI) # Standardize ROI
    ROIs = [imageROI, labelROI]
    if segmentation is not False:
        segmentationROI = segmentation[:,midSlice-int(ROI_Height/2):midSlice+int(ROI_Height/2),ROI_startWidth:ROI_endWidth]    
        ROIs = [imageROI, labelROI, segmentationROI]
    return ROIs

def read_data(image_path, label_path, segmentation_path=False):
    '''
    This function read the dataset given image, label and segmentation path

    '''
    # Data list
    imageList = natsorted(os.listdir(image_path))
    labelList = natsorted(os.listdir(label_path))
    if segmentation_path is False:
        images, labels = [], []
        for n in tqdm(range(len(labelList))):
            image, label = read_image(image_path + imageList[n],
                                      label_path + labelList[n])
            # Extract ROI
            imgROI, lblROI = select_ROI(image, label)
            # Binary Label
            lblROI[lblROI>0] = 1
            # Append lists
            images.append(imgROI)
            labels.append(lblROI)
            # Numpy array from lists
        images, labels = np.array(images).astype(np.float32), np.array(labels).astype(np.uint8)
        dataset = [images, labels]
    else:
        images, labels, segmentations = [], [], []
        segmentationList = natsorted(os.listdir(segmentation_path))
        for n in tqdm(range(len(segmentationList))):
            image, label, segmentation = read_image(image_path + imageList[n],
                                                    label_path + labelList[n],
                                                    segmentation_path + segmentationList[n])
            # Extract ROI
            imgROI, lblROI, segROI = select_ROI(image, label, segmentation)
            # Binary masks
            lblROI[lblROI>0] = 1
            segROI[segROI>0] = 1
            # Append lists
            images.append(imgROI)
            labels.append(lblROI)
            segmentations.append(segROI)
            # Numpy array from lists
        images, labels, segmentations = np.array(images).astype(np.float32), np.array(labels).astype(np.uint8), np.array(segmentations).astype(np.uint8)
        dataset = [images, labels, segmentations]
    return dataset

def interpolate_DSC(patchwiseDSC, imageROI, r=8):
    '''
    This function interpolate the patch-wise DSC scores to voxel-wise DSC score

    Parameters
    ----------
    patchwiseDSC : predicted patchwiseDSC 3D-numpy array from REGNET
    imageROI : 3D-numpy array
    r : size of the patch (rxrxr), The default is 8.

    Returns
    -------
    pDSC : voxelwise DSC (3D-numpy array)

    '''
    pDSC = np.zeros_like(imageROI)
    for d in range(patchwiseDSC.shape[0]):
        for h in range(patchwiseDSC.shape[1]):
            for w in range(patchwiseDSC.shape[2]):
                pDSC[r*d:r*(1+d),r*h:r*(1+h),r*w:r*(1+w)]=patchwiseDSC[d,h,w]
    return pDSC

# def interpolate_DSC(patchwiseDSC, imageROI):
#     '''
#     This function resize the patch-wise DSC score to ROI size

#     '''
#     from skimage.transform import resize
#     size = imageROI.shape
#     capI = resize(patchwiseDSC, size)
#     capI = np.maximum(capI,0)
#     resizedImage = (capI - capI.min()) / (capI.max() - capI.min())
#     return resizedImage

def surface_edge(volume):
    '''
    This function generates edges per slice given 3D label image

    Parameters
    ----------
    volume : 3D numpy array (label image)

    '''
    import scipy as sp
    import scipy.ndimage
    surface = np.zeros_like(volume)
    for i in range(len(volume)):
        imax=(sp.ndimage.maximum_filter(volume[i],size=3)!=volume[i])
        imin=(sp.ndimage.minimum_filter(volume[i],size=3)!=volume[i])
        icomb=np.logical_or(imax,imin)
        edges=np.where(icomb,volume[i],0)
        edges[edges>0]=1
        surface[i] = edges[:]
    return surface

def image_color(image, label):
    '''
    This function overlay colored label edge on the grayscale image

    Parameters
    ----------
    image : 3D numpy array
    label : 3D numpy array (label edge image)

    '''
    label1 = np.where(label>0, 1, image)
    label0 = np.where(label>0, 0, image)
    coloredImage = np.stack([label0, label1, label0], axis=-1)
    return coloredImage

def generate_real_samples(trainA, trainB, sample):
    '''
    This function generates real samples for discriminator training

    Parameters
    ----------
    trainA : Input image
    trainB : Target image
    sample : Sample IDs

    '''
    st, en = sample
    X1, X2 = trainA[st:en], trainB[st:en]
    y = np.ones(((en-st), 10, 6, 15, 1)).astype(np.float32)
    return [X1, X2], y

def generate_fake_samples(g_model, samples):
    '''
    This function generates fake samples for discriminator training

    Parameters
    ----------
    g_model : RECNET model
    samples : Sample IDs

    '''
    X = (g_model.predict(samples)).astype(np.float32)
    y = np.zeros((len(X), 10, 6, 15, 1)).astype(np.float32)
    return X, y
 
def train_GANRECNET(d_model, g_model, gan_model, xtrain, ytrain, xval, yval, n_epochs=100, n_batch=4, weightPath='./modelWeights/demo_knee_train_RECNET.hdf5'):
    '''
    This function trains the GANREC-NET model and save model weights

    Parameters
    ----------
    d_model : Discriminator mode
    g_model : RECNET model
    gan_model : GANREC-NET model
    xtrain : Input image (train)
    ytrain : Target image (train)
    xval : Input image (validation)
    yval : Target image (validation)
    n_epochs : Number of epoch
    n_batch : Batch size
    weightPath : Model weight path
    
    '''
    print('\n--------------------------------------')
    print(str('nSample:{:d} \tnTrain:{:d} \tnValidation:{:d}'.format(len(xtrain)+len(xval),len(xtrain),len(xval))))
    print('image shape: ', xtrain.shape[1:])
    print('\nTraining GANREC-NET..............')
    d1loss,d2loss,gloss,vloss,lloss = [],[],[],[],float('inf')
    count=1
    for i in range(n_epochs):
        if n_batch>len(xtrain):
            n_batch = len(xtrain)
        for j in range(0,len(xtrain),n_batch):
            st, en = j, j+n_batch
            if en>len(xtrain):
                en = len(xtrain)
            [X_realA, X_realB], y_real = generate_real_samples(xtrain, ytrain, (st,en))
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA)
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        vpred = gan_model.predict(xval, batch_size=n_batch)[1]
        v_loss = np.mean(np.abs(yval - vpred))   
        d1loss.append(d_loss1)
        d2loss.append(d_loss2)
        gloss.append(g_loss)
        vloss.append(v_loss)
        print('>epoch-%d, loss_real-[%.3f] loss_fake-[%.3f] g-[%.3f] v-[%.3f]' % (i+1, d_loss1, d_loss2, g_loss, v_loss))
        if v_loss<lloss:
            g_model.save_weights(weightPath)
            print('validation loss improved from [%.3f] to [%.3f]. Saving the model weights in the weightPath' % (lloss, v_loss))
            lloss = v_loss
            count+=1
    print('Training done!\n')
    return d1loss, d2loss, gloss, vloss

def plot_trainLoss(real_loss, fake_loss, GAN_loss, val_loss):
    '''
    This function plots the training and validation losses per epoch

    Parameters
    ----------
    real_loss : Discriminator loss for real samples
    fake_loss : Discriminator loss for fake samples
    GAN_loss : GANREC-NET train loss
    val_loss : GANREC-NET validation loss

    '''
    x = np.arange(1, len(real_loss)+1)
    fig=plt.figure(figsize=(12,5.5))
    fig.suptitle('GAN Losses',fontsize=20)
    ax1 = fig.add_subplot(121)
    ax1.plot(x, GAN_loss)
    ax1.plot(x, val_loss)
    ax1.legend(['train_loss', 'validation_loss'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Combined Loss',fontsize=16)
    ax2 = fig.add_subplot(122)
    ax2.plot(x, real_loss)
    ax2.plot(x, fake_loss)
    ax2.legend(['discriminator_loss','generator_loss'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('D & G Loss',fontsize=16)
    
def plot_checkRECNET(xval, yval, weightPath='./modelWeights/demo_knee_train_RECNET.hdf5'):
    '''
    This function plots sample images from validation set
    to check the trained RECNET model

    Parameters
    ----------
    xval : Input validation image
    yval : Target validation image
    weightPath : Path of the RECNET weights

    '''
    tf.keras.backend.clear_session()
    img_shape = xval.shape[1:]
    nsample, slc = np.random.randint(0, len(xval)), int(xval.shape[1]/2)
    conditionedImage, targetImage = np.squeeze(xval[nsample][slc]), np.squeeze(yval[nsample][slc])
    RECNET = models.define_RECNET(img_shape)
    untrainedImage = np.squeeze(RECNET.predict(np.expand_dims(xval[nsample], axis=0), batch_size=1))[slc]
    RECNET.load_weights(weightPath)
    trainedImage = np.squeeze(RECNET.predict(np.expand_dims(xval[nsample], axis=0), batch_size=1))[slc]
    # Image plots
    fig=plt.figure(figsize=(12,5.5))
    ax1 = fig.add_subplot(221)
    ax1.imshow(conditionedImage, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Conditioned Image',fontsize=16)
    ax2 = fig.add_subplot(222)
    ax2.imshow(targetImage, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Target Image',fontsize=16)
    ax3 = fig.add_subplot(223)
    ax3.imshow(untrainedImage, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Untrained RECNET',fontsize=16)
    ax4 = fig.add_subplot(224)
    ax4.imshow(trainedImage, cmap='gray')
    ax4.axis('off')
    ax4.set_title('Trained RECNET',fontsize=16)
    plt.tight_layout()
    
def predictions_RECNET(xtrain, ytrain=[], weightPath='./modelWeights/demo_knee_train_RECNET.hdf5'):
    '''
    This function generates predictions of RECNET given input conditioned
    images and prints the training errors
    
    Parameters
    ----------
    xtrain : Input conditioned image (5-D numpy array)
    ytrain : Target image (5-D numpy array)
    weightPath : Path of the RECNET weights
    
    Returns
    -------
    MAE (Mean Absolute Error), L2-norm
    
    '''
    if not os.path.exists(weightPath):
        raise Exception('RECNET model not found. Please train the RECNET model first using train_GANREC-NET.py')
    tf.keras.backend.clear_session()
    predictionsRECNET = []
    RECNET = models.define_RECNET(xtrain.shape[1:])
    weightPath = './modelWeights/demo_knee_train_RECNET.hdf5'
    RECNET.load_weights(weightPath)
    print('\nPredicting using RECNET..............')
    for n in range(len(xtrain)):
        predictionsRECNET.append(np.squeeze(RECNET.predict(np.expand_dims(xtrain[n], axis=0), batch_size=1)))
    predictionsRECNET = np.array(predictionsRECNET)
    print('\nPredicting done!')
    # recon error
    if len(ytrain)>0:
        print('\nCalculating errors..............')
        MAE = np.mean(np.abs(np.squeeze(ytrain)-predictionsRECNET))
        L2 = []
        for n in range(len(predictionsRECNET)):
            L2.append(np.linalg.norm(np.squeeze(ytrain[n]) - predictionsRECNET[n]))
        L2 = np.mean(L2)
        print('\n-----------------------')
        print(str('Train MAE = {:0.3f}'.format(MAE)))
        print(str('Train L2-norm = {:0.3f}'.format(L2)))
        print('-----------------------')
    return predictionsRECNET

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """Altered Sorensen–Dice coefficient with epsilon for smoothing."""
    y_true_flatten = np.asarray(y_true).astype(np.bool_)
    y_pred_flatten = np.asarray(y_pred).astype(np.bool_)
    if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
        Dice=1.0
    else:
        Dice=(2. * np.sum(y_true_flatten * y_pred_flatten)) /\
            (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)
    return Dice

def dice_patch(y_true, y_pred, r=8):
    '''
    This function generates patch-wise DSC

    '''
    do, ho, wo = y_true.shape
    d, h, w = int(do/r), int(ho/r), int(wo/r)
    dice_mat = np.zeros((d, h, w))
    for ds in range(d):
        for hs in range(h):
            for ws in range(w):
                std, sth, stw = r*ds, r*hs, r*ws
                end, enh, enw = r*(1+ds), r*(1+hs), r*(1+ws)
                if do-end<d:
                    end = do
                if ho-enh<h:
                    enh = ho
                if wo-enw<w:
                    enw = wo
                true = y_true[std:end, sth:enh, stw:enw]
                pred = y_pred[std:end, sth:enh, stw:enw]
                dice_mat[ds,hs,ws] = dice_coefficient(true, pred)
    return dice_mat

def patchwise_DSC(labels, segmentations, r=8):
    '''
    This function generates patch-wise DSC for the dataset

    Parameters
    ----------
    labels : ground truths
    segmentations : segmentation results
    
    '''
    dices = []
    for n in tqdm(range(len(labels))):
        dices.append(dice_patch(labels[n], segmentations[n], r))
    return np.array(dices)

def train_REGNET(REGNET_model, xtrain, ytrain, xval, yval, n_epochs=100, n_batch=4, weightPath='./modelWeights/demo_knee_train_REGNET.hdf5'):
    '''
    This function trains the REGNET model and save model weights

    Parameters
    ----------
    REGNET_model : REGNET model
    xtrain : Train stacked image
    ytrain : Train patch-wise DSC
    xval : Validation stacked image
    yval : Validation patch-wise DSC
    n_epochs : Number of epoch
    n_batch : Batch size
    weightPath : Model weight path

    Returns
    -------
    None.

    '''
    print('\n--------------------------------------')
    print(str('nSample:{:d} \tnTrain:{:d} \tnValidation:{:d}'.format(len(xtrain)+len(xval),len(xtrain),len(xval))))
    print('image shape: ', xtrain.shape[1:])
    print('\nTraining REGNET..............')
    checkpoint = ModelCheckpoint(weightPath, monitor='val_loss', verbose=1, 
                              save_best_only=True, mode='min')
    stop = EarlyStopping(monitor='val_loss', patience=50, mode="min")
    callbacks_list = [checkpoint, stop] 
    history = REGNET_model.fit(xtrain, ytrain,
                               epochs = n_epochs,
                               batch_size = n_batch,
                               validation_data = (xval, yval),
                               callbacks = callbacks_list, verbose=1)
    print('Training done!\n')
    # Loss plots
    plt.figure()
    plt.title('Loss plots')
    plt.plot(history.history['val_loss'], label='validation-loss')
    plt.plot(history.history['loss'], label='training-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()