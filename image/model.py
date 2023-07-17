import os
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
# class_count = len(list(train_gen.class_indices.keys()))

class Models(ABC):
    """
    Abstract base class that defines and creates model.
    """
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass


@dataclass
class EfficientNetB3(Models):
    train_gen: Any=None
    def define_model(self):
        # print(type(self.train_gen))
        class_count = len(list(self.train_gen.class_indices.keys()))
        img_shape = (224,224,3)
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
# base_model.trainable = False

        model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                    bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])
        return model
    
    def create_model(self):
        model = self.define_model()
        model.summary()
        return model
        
  
