import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE


@dataclass
class Dataset:
    """Dataset Class used to hold train-test split data which can be passed to 
    ML Algorithm
    """
    x_train: np.ndarray = None
    y_train: np.array = None
    x_test: np.ndarray = None
    y_test: np.array = None


@dataclass
class Preprocessing:

      # data = pd.read_csv(
        #     '/home/ali/Desktop/FYP/MLflow/mlops_fyp/dataset/train/Churn_Modelling.csv')


    def preprocess(self, test_size_percent,train_data):
        """This function is of class Preprocessing and is used for Preprocssing the data. It takes the only one argument
        train-test split size.

        Returns:
            Dataset: Dataset having data splitted as x_train,y_train,x_test,y_test
        """

        

        data, labels = train_data.iloc[:, 0:-1], train_data.iloc[:, -1]

        # Checking the missing values in data and remove them from the data

        # Feature Engineering
        
        data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

        

        le = LabelEncoder()

        data['Geography'] = le.fit_transform(data['Geography'])
        data['Gender'] = le.fit_transform(data['Gender'])

        
        
        x_resampled, y_resampled = SMOTE().fit_resample(data, labels)
        
        # print(x_resampled)
        print(x_resampled.mean())
        print(x_resampled.std())
        scalar = StandardScaler()
        x_resampled = scalar.fit_transform(x_resampled)
        # print(scalar.mean_)
        # print(scalar.var_)

        x_train, x_test, y_train, y_test = train_test_split(
            x_resampled, y_resampled, test_size=test_size_percent, random_state=42)

        
        return Dataset(x_train, y_train, x_test, y_test)
    
