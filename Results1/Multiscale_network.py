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


def multiscale_network(): 

    
    # Input 8x8
    input_img = tf.keras.Input(shape=(8, 8, 1))
    

    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(input_img)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    
    x = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2))(x)
    
    x16 = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same',name='x16')(x)
    #till 16x16
    
    
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x16)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    
    x=tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2))(x)
    
    x32 = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same',name='x32')(x)
    #till 32x32
    
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same') (x32)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same')(x)
    
    x=  tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2))(x)
    
    x64 = tf.keras.layers.Conv2D(1, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                      padding='same',name='x64')(x)
    

    model= tf.keras.Model(inputs=input_img, outputs=[x16, x32, x64])
    
    model.compile(optimizer='adam', loss={'x16':'mean_absolute_error','x32':'mean_absolute_error',
                                                    'x64':'mean_absolute_error'})
    
    return model