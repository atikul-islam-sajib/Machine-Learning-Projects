# import tensorflow as tf
import keras.utils as image
from keras.models import save_model
import numpy as np
from tensorflow import keras
##### Import all necessity functions for Machine Learning #####
import sys
import math
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import scipy as shc
import warnings
import zipfile
# import cv2
import os
import random
from collections import Counter
from functools import reduce
from itertools import chain
from keras.preprocessing import image

##### Import all necessity functions for Neural Network #####
import tensorflow as tf
from keras.utils import plot_model
# from tensorflow.keras.regularizers import L1, L2, L1L2
# from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
# from tensorflow.keras.initializers import HeNormal, HeUniform, GlorotNormal, GlorotUniform
# from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, hinge, MSE, MAE, Huber
import keras.utils as image
import keras

from flask import Flask, render_template, request, session, flash, redirect, url_for
from werkzeug.utils import secure_filename



app = Flask(__name__, template_folder='template',  static_folder='static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')

def index():
    return  render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])

def predict():
    from keras.models import save_model
    model = keras.models.load_model('C:/Users/DELL/PycharmProjects/pythonProject5/brain_tumor.h5', compile=False)

    target = os.path.join(APP_ROOT, 'static/')

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)


    Image = image.load_img("C:/Users/DELL/PycharmProjects/pythonProject5/static/Te-pi_0011.jpg", target_size = (150, 150))
    Image_array_ = np.asarray(Image)
    Image_array_ = image.img_to_array(Image)
    Image_array_numpy = (Image_array_/255)
    test_data = np.expand_dims(Image_array_numpy, axis = 0)
    predicted_ =  model.predict(test_data)
    predicted_ = np.argmax(predicted_, axis = 1)
    return predicted_

    if predicted_[0] == 0:
        print('Normal')
    elif predicted_[0] == 1:
        print('Ulcer')
    elif predicted_[0] == 2:
         print('Polyps')
    else:
        print('Esophagitis')

app.run(debug=True)

from keras.models import save_model
# model = keras.models.load_model('C:/Users/DELL/PycharmProjects/pythonProject5/brain_tumor.h5', compile=False)
#
#
# Image = image.load_img('C:/Users/DELL/PycharmProjects/pythonProject5/Te-gl_0013.jpg', target_size = (150, 150))
# Image_array_ = image.img_to_array(Image)
# Image_array_numpy = (Image_array_/255)
# test_data = np.expand_dims(Image_array_numpy, axis = 0)
# predicted_ =  model.predict(test_data)
# predicted_ = np.argmax(predicted_, axis = 1)
#

