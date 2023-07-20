
import hydra
from src.model import RandomForest,XGBoost
from src.data import Preprocessing
from src.experiment_tracking import ModelSelection, MLFlowTracker
from src.train import Training
from hydra.core.config_store import ConfigStore
from config_dir.config_struct import ChurnConfig
import pandas as pd
from omegaconf import OmegaConf


cfg = OmegaConf.load('./config_dir/config.yaml')

class Retrain():
    def __init__(self,data_path):
        self.data_path= data_path
    
    def retrain_model(self):
        print("Now")
        # print(cfg)
        print(cfg.churn_paths.mlflow_tracking_uri)
        tracker = MLFlowTracker(cfg.churn_names.experiment_name,
                                cfg.churn_paths.mlflow_tracking_uri)
        
        
        #Data Merger
        
        previous_data_read = pd.read_csv(cfg.churn_paths.train_dir)
        new_data_read = pd.read_csv(self.data_path)
        
        
        merged_data = pd.concat([previous_data_read,new_data_read],axis='rows')
        
        
        preprocessing = Preprocessing()
        pre_processed_dataset = preprocessing.preprocess(
            cfg.churn_params.train_test_split,merged_data)
        
        model= RandomForest()
        # model = RandomForest(cfg.params.n_estimators, cfg.params.criterion)
        model = model.create_model()
        # tracker.log_params(cfg.params.n_estimators,cfg.params.criterion)
        best_selected_model = Training(
            model, pre_processed_dataset, tracker, cfg.churn_names.metric_name,cfg.churn_paths.fig_path)
        best_selected_model.train()
        
        
        print("Model Successfully retrained")