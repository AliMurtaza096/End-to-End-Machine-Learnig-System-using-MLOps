import os
# import hydra
import numpy as np
import mlflow
from image.data import Preprocessing
# from src.config_dir.config_struct import ChurnConfig
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import json


from PIL import Image




class DiseasePredict:
    
    """_summary_
    """
    
    def __init__(self,model_artifact_uri):
        
        """_summary_

        Args:
            model_artifact_uri (_type_): _description_
        """
        
        self.model_artifact_uri =model_artifact_uri
        self.labels = ['NORMAL','PNEMONIA']

    def load_model(self):
        loaded_model = mlflow.pyfunc.load_model(self.model_artifact_uri)
        return loaded_model
        
    def predict(self,image_path,loaded_model):
        
        """_summary_

        Returns:
            _type_: _description_
        """
        
        
        # print(image_path)
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match the model's input size
        img = img.convert('RGB')  # Convert to RGB if needed
        img = np.array(img)  # Convert PIL image to numpy array
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        

        
        # file1 = open('/home/ali/Desktop/FYP/MLflow/mlops_fyp/src/schema.json','r')
        # schema =json.load(file1)
        # for row in zip(schema,test_data_values):
        #     schema_key =row[0]
            
        #     schema_instance_mean = schema[schema_key]['mean']
        #     schema_instance_std = schema[schema_key]['std']
            
        #     test_data_instance = int(row[1])
            
        #     transformed_value= (test_data_instance-schema_instance_mean) / schema_instance_std
            
        #     transformed_data.append(transformed_value)
            
        # transformed_data = np.array(transformed_data)
        # transformed_data = transformed_data.reshape(-1,10)
        # scaler  =StandardScaler()
        
        # test_transformed = scaler.fit_transform(test_data_values)
      
        
        # Load model as a PyFuncModel.
        
        # print(loaded_model)
        # # Predict on a Pandas DataFrame.
        prediction = loaded_model.predict(img)
        y_pred = np.argmax(prediction, axis=1)
        
        return self.labels[y_pred[0]]



