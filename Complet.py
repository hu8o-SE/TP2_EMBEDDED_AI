#################################################################################################################
# IMPORTS AND SETTINGS 
#################################################################################################################

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import cv2
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report
from skimage import exposure
import zipfile
from PIL import Image
import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Dense, Flatten, Dropout, Activation

print(tf.__version__)

#################################################################################################################
# DATASET EXPLORATION 
#################################################################################################################

path_ds = '/Users/hugo/Desktop/TPs_IA_embarquée/TP2'
print(os.listdir(path_ds))
print( os.listdir(os.path.join(path_ds,'train')))

#################################################################################################################
# DATASET LOADING AND PREPROCESSING
#################################################################################################################

# Setting variables for later use
data = [] # ici je vais stocker les images
labels = [] # ici les labels
classes = 43 # c'est le nombre de classes dans le dataset
n_learned_classes = classes # alias pour manipuler plus facielement le nombre de classes


# Retrieving the images and their labels
for i in range(n_learned_classes): # Boucle sur les classes
    path = os.path.join(path_ds,'train',str(i))
    images = os.listdir(path)
    #print ('Classe : ' + str(i), len(images)) # on affiche le nombre d'images par classes pour vérifier la cohérence du dataset

    for a in images: # Boucle sur les images
        try:
            image = Image.open(path + '/' + a  ) # on ouvre
            image = image.resize((30,30)) # On redimension l'image en 30x30 pixels
            # ces deux lignes si-dessous, on va les utiliser plus tard
            image = np.array(image) # on convertit en tableau numpy HxLxC donc 30x30x3
            # placer ici une équalisation de l'histogramme de l'image
            image =  exposure.equalize_adapthist(image, clip_limit=0.1)# On vient rendre les images plus lisibles
            data.append(image)# On ajoute l'image dans la liste data
            labels.append(i)#on ajoute le label dans la liste labels
        except:
            print("Error loading image") # Si une image ne peut pas être chargée, message d'erreur


# Converting lists into numpy arrays
data = np.array(data)#On convertit la liste en tableau numpy car mieux pour ccn
labels = np.array(labels)#idem
print(data.size)
print(labels.size)

