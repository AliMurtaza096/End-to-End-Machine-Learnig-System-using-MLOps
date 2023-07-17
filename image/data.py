import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
import  matplotlib.pyplot as plt 
from typing import Any


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator




# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Dataset:
    """Dataset Class used to hold train-test split data which can be passed to 
    ML Algorithm
    """
    train_gen: Any = None
    valid_gen: Any = None
    test_gen: Any = None



@dataclass
class Preprocessing:

      # data = pd.read_csv(
        #     '/home/ali/Desktop/FYP/MLflow/mlops_fyp/dataset/train/Churn_Modelling.csv')
    data_path:str
    test_size_percent:float


    def __load_and_split(self):
        """This function is of class Preprocessing and is used for Preprocssing the data. It takes the only one argument
        train-test split size.

        Returns:
            Dataset: Dataset having data splitted as x_train,y_train,x_test,y_test
        """
        filepaths = []
        labels = []

        folds = os.listdir(self.data_path)
        for fold in folds:
            foldpath = os.path.join(self.data_path, fold)
            filelist = os.listdir(foldpath)
            for file in filelist:
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)
        
        Fseries = pd.Series(filepaths, name= 'filepaths')
        Lseries = pd.Series(labels, name='labels')
        df = pd.concat([Fseries, Lseries], axis= 1)
        
        train_df, dummy_df = train_test_split(df,  test_size=self.test_size_percent, shuffle= True, random_state= 123)

        # valid and test dataframe
        valid_df, test_df = train_test_split(dummy_df,  test_size=self.test_size_percent, shuffle= True, random_state= 123)

        return train_df, dummy_df, valid_df, test_df
    def data_augmentation(self):

        train_df,dummy_df,valid_df,test_df= self.__load_and_split()
        batch_size = 16
        img_size = (224, 224)
        channels = 3
        img_shape = (img_size[0], img_size[1], channels)

        # Recommended : use custom function for test data batch size, else we can use normal batch size.
        ts_length = len(test_df)
        test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
        test_steps = ts_length // test_batch_size

        # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
        def scalar(img):
            return img

        tr_gen = ImageDataGenerator(preprocessing_function= scalar)
        ts_gen = ImageDataGenerator(preprocessing_function= scalar)
        # print(type(tr_gen))
        train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

        valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

        # Note: we will use custom test_batch_size, and make shuffle= false
        test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= test_batch_size)
        
        g_dict = train_gen.class_indices      # defines dictionary {'class': index}
        classes = list(g_dict.keys())
        print(classes)     
        return Dataset(train_gen,valid_gen,test_gen)

# new_= Preprocessing(test_size_percent=0.3)
# train_gen,valid_gen,test_gen = new_.data_augmentation()
# model= EfficientNetB3(train_gen)
# print(model.create_model())

