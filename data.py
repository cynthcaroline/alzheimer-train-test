import numpy as np
import os.path
import time
import sys
import time
import pyprind
import cv2
from imutils import face_utils

import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import bz2
import os
from PIL import Image

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        list_files = os.listdir(os.path.join(path, i))
        if len(list_files) > 10:
            for f in list_files:
                # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg':
                    metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def preprocess_image(path,size=(150,150)):
    img = cv2.imread(path)
    img = cv2.resize(img,size)
    img = (img / 255.)

    return img

def load_dataset(path,test_size=0.2):
    print('Load metadata...')
    metadata = load_metadata(path)
    print(len(metadata))
    
    
    img = np.empty((len(metadata), 150, 150, 3))
    labels = []
    
    bar = pyprind.ProgBar(len(metadata),bar_char='█')
    print('Read image...')
    i = 0
    for img_meta in metadata:
        image = preprocess_image(img_meta.image_path())
        label = img_meta.name
        print(img_meta.name)
        img[i] = image
        labels.append(label)
        i += 1
        bar.update()
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(categories='auto',sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    x = img
    y = onehot_encoded

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = test_size, random_state = None)    

    return xTrain, xTest, yTrain, yTest, label_encoder

if  __name__ == '__main__':
     xTrain, xTest, yTrain, yTest, label_encoder  = load_dataset('data/mri',0.2)
     
     print(xTrain.shape)
     print(yTrain.shape)
     print(xTest.shape)
     print(yTest.shape)

     print(yTrain[0])
       
     inverted = label_encoder.inverse_transform([np.argmax(yTrain[0])])[0]
     print(inverted)
