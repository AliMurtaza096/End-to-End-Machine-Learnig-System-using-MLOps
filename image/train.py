from data import Dataset
from model import EfficientNetB3
from experiment_tracking import MLFlowTracker,ModelSelection
import tensorflow as tf

from keras.optimizers import Adam, Adamax

import os
class Training:
    def __init__(self,model:EfficientNetB3, data:Dataset,
                tracker:MLFlowTracker,metric_name:str,
                batch_size:int,epochs:int,learning_rate:float) -> None:
        self.model = model
        self.data = data
        self.tracker = tracker
        self.metric_name = metric_name
        self.batch_size= batch_size
        self.epochs=epochs
        self.learning_rate = learning_rate
        
        
    def train(self) -> ModelSelection:
        artifact = self.model.compile(Adamax(learning_rate= self.learning_rate), loss= 'categorical_crossentropy', metrics= [self.metric_name])
        print(type(self.model))
        # self.tracker.log()
        history = self.model.fit(x= self.data.train_gen, epochs= self.epochs, verbose= 1, validation_data= self.data.valid_gen, 
                    validation_steps= None, shuffle= False)
        # print(type)

       
        
        # model_weights = [weight.numpy() for weight in self.model.weights]

# Save the model using mlflow.tensorflow.save_model()
        # mlflow.tensorflow.save_model(model, "model_dir", signature="serving_default", input_example=model_weights[0])
        # self.model.save(path)
        # self.tracker.save_model(self.model,path)
        # tf.saved_model(self.model,path)
        print(self.tracker.find_best_model(self.metric_name))

        return ModelSelection(self.tracker.find_best_model(self.metric_name))