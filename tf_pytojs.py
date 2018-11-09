import tensorflowjs as tfjs
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import load_model
from keras.optimizers import SGD,adam, RMSprop
from keras.utils import np_utils
import keras.backend as K
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import sklearn 
import tensorflow
from sklearn.model_selection import train_test_split
from PIL import Image

def py_to_js():
	load_path=r"20180404_4class.hdf5"
	model=load_model(load_path)
	tfjs_target_dir=r"C:\Users\justBaloney\Final Year Project"
	tfjs.converters.save_keras_model(model, tfjs_target_dir)
