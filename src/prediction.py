import os
import hydra
import numpy as np
import mlflow
from src.data import Preprocessing
# from config_dir.config_struct import ChurnConfig
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import json



class ChurnPredict:
    
    """_summary_
    """
    
    
    def __init__(self,model_artifact_uri,**kwargs):
        
        """_summary_

        Args:
            model_artifact_uri (_type_): _description_
        """
        
        self.model_artifact_uri =model_artifact_uri
        self.dataa = kwargs
    def predict(self):
        
        """_summary_

        Returns:
            _type_: _description_
        """
        
        test_data_values = list(self.dataa.values())
        transformed_data =[]
        
        file1 = open('./src/schema.json','r')
        schema =json.load(file1)
        for row in zip(schema,test_data_values):
            schema_key =row[0]
            
            schema_instance_mean = schema[schema_key]['mean']
            schema_instance_std = schema[schema_key]['std']
            
            test_data_instance = int(row[1])
            
            transformed_value= (test_data_instance-schema_instance_mean) / schema_instance_std
            
            transformed_data.append(transformed_value)
            
        transformed_data = np.array(transformed_data)
        transformed_data = transformed_data.reshape(-1,10)
        # scaler  =StandardScaler()
        
        # test_transformed = scaler.fit_transform(test_data_values)
      
        
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(self.model_artifact_uri)
        # Predict on a Pandas DataFrame.
        prediction = loaded_model.predict(transformed_data)
        
        return prediction
    
    def batch_predict(self):
        self.dataa = self.dataa['data']
        print(self.dataa)
        
        self.dataa.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

      

        le = LabelEncoder()

        self.dataa['Geography'] = le.fit_transform(self.dataa['Geography'])
        self.dataa['Gender'] = le.fit_transform(self.dataa['Gender'])
        scaler  =StandardScaler()
        
        test_transformed = scaler.fit_transform(self.dataa)
      
        
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(self.model_artifact_uri)
        # Predict on a Pandas DataFrame.
        prediction = loaded_model.predict(test_transformed)
        print(prediction)
        self.dataa['Results']  = prediction

        return self.dataa