#Imports
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from skimage.metrics import structural_similarity as ssim
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from datetime import datetime


def autoencoder(z_dim): 
    
    input_img = tf.keras.Input(shape=(64, 64, 1))
    

    x = tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(input_img)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
        
    x = tf.keras.layers.Conv2D(8, (4, 4), padding='same')(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    
    x = tf.keras.layers.Conv2D(16, (5, 5), padding='same')(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    
    
    x=tf.keras.layers.Dense(z_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    
    
    
    x=tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    
    x=tf.keras.layers.Reshape((8,8,16))(x)
    
    
    
    x = tf.keras.layers.Conv2D(16, (5, 5), padding='same')(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    
    
    
    x = tf.keras.layers.Conv2D(8, (4, 4), padding='same')(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    
    
    
    x = tf.keras.layers.Conv2D(4, (3, 3), padding='same')(x)
    
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    
    output = tf.keras.layers.Conv2D(1, (3, 3), padding='same',name='output',activation='sigmoid')(x)
    
    model= tf.keras.Model(inputs=input_img, outputs=[output])
    adamm= tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(optimizer=adamm, loss={'output':'mean_squared_error'})
    
    return model

