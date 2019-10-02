from keras.applications import VGG16
import os
import numpy as np

from keras import models
from keras import layers
from keras.utils.vis_utils import plot_model

import pyprind 


conv_base = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3))


def vgg16_finetuned(num_classes):
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model




if __name__ == '__main__':
    net = vgg16_finetuned(11)
    net.summary()
    plot_model(conv_base, to_file='data/result/base_model.png', show_shapes=True, show_layer_names=True)
    plot_model(net, to_file='data/result/tuned_model.png', show_shapes=True, show_layer_names=True)
