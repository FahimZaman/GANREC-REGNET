#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:13:29 2021

@author: fazaman
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv3D, Conv3DTranspose
from keras.layers import MaxPooling3D
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam

def define_RECNET(image_shape):
    '''
    This function creates the REC-NET model with given image shape

    '''
    in_image = Input(shape=image_shape)
    
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (in_image)
    c1 = LeakyReLU(alpha=0.2)(c1)
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (c1)
    c1 = LeakyReLU(alpha=0.2)(c1)
    p1 = MaxPooling3D(pool_size=(2,2,2)) (c1)
    
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (p1)
    c2 = LeakyReLU(alpha=0.2)(c2)
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (c2)
    c2 = LeakyReLU(alpha=0.2)(c2)
    p2 = MaxPooling3D(pool_size=(2,2,2)) (c2)
    
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (p2)
    c3 = LeakyReLU(alpha=0.2)(c3)
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (c3)
    c3 = LeakyReLU(alpha=0.2)(c3)
    p3 = MaxPooling3D(pool_size=(2,2,2)) (c3)
    
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (p3)
    c4 = LeakyReLU(alpha=0.2)(c4)
    c4 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal') (c4)
    c4 = LeakyReLU(alpha=0.2)(c4)
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (c4)
    c4 = LeakyReLU(alpha=0.2)(c4)
        
    c5 = Conv3DTranspose(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(c4)
    c5 = Concatenate()([c5, c3])
    c5 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal') (c5)
    c5 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal') (c5)
    
    c6 = Conv3DTranspose(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(c5)
    c6 = Concatenate()([c6, c2])
    c6 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal') (c6)
    c6 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal') (c6)
    
    c7 = Conv3DTranspose(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(c6)
    c7 = Concatenate()([c7, c1])
    c7 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal') (c7)
    out_image = Conv3D(1, 3, padding='same', activation='tanh', kernel_initializer='he_normal') (c7)
    model = Model(in_image, out_image)
    return model

def define_discriminator(image_shape):
    '''
    This function creates the discriminator for REC-NET model with given image shape

    '''
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([in_src_image, in_target_image])
    
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (merged)
    c1 = LeakyReLU(alpha=0.2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (c1)
    c1 = LeakyReLU(alpha=0.2)(c1)
    p1 = MaxPooling3D(pool_size=(2,2,2)) (c1)
    
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (p1)
    c2 = LeakyReLU(alpha=0.2)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (c2)
    c2 = LeakyReLU(alpha=0.2)(c2)
    p2 = MaxPooling3D(pool_size=(2,2,2)) (c2)
    
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (p2)
    c3 = LeakyReLU(alpha=0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (c3)
    c3 = LeakyReLU(alpha=0.2)(c3)
    p3 = MaxPooling3D(pool_size=(2,2,2)) (c3)
    
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (p3)
    c4 = LeakyReLU(alpha=0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (c4)
    c4 = LeakyReLU(alpha=0.2)(c4)
    p4 = MaxPooling3D(pool_size=(2,2,2)) (c4)
    
    patch_out = Conv3D(1, 3, padding='same', activation='sigmoid', kernel_initializer='he_normal') (p4)
    model = Model([in_src_image, in_target_image], patch_out)
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_GANRECNET(g_model, d_model, image_shape):
    '''
    This function creates the GANREC-NET model with given image shape

    Parameters
    ----------
    g_model : RECNET model
    d_model : Discriminator mode
    
    '''
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, 
                  loss_weights=[1,100])
    return model

def define_REGNET(image_shape):
    '''
    This function creates the REG-NET model with given image shape

    '''
    in_image = Input(shape=image_shape)
        
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (in_image)
    c1 = LeakyReLU(alpha=0.2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal') (c1)
    c1 = LeakyReLU(alpha=0.2)(c1)
    p1 = MaxPooling3D(pool_size=(2,2,2)) (c1)
    
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (p1)
    c2 = LeakyReLU(alpha=0.2)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal') (c2)
    c2 = LeakyReLU(alpha=0.2)(c2)
    p2 = MaxPooling3D(pool_size=(2,2,2)) (c2)
    
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (p2)
    c3 = LeakyReLU(alpha=0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal') (c3)
    c3 = LeakyReLU(alpha=0.2)(c3)
    p3 = MaxPooling3D(pool_size=(2,2,2)) (c3)
    
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (p3)
    c4 = LeakyReLU(alpha=0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal') (c4)
    c4 = LeakyReLU(alpha=0.2)(c4)
        
    patch_out = Conv3D(1, 3, padding='same', activation='sigmoid', kernel_initializer='he_normal') (c4)
    model = Model(in_image, patch_out)
    opt = Adam(learning_rate=0.00004, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model