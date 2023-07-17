import warnings
import hydra
from src.model import RandomForest,XGBoost
from src.data import Preprocessing
from src.experiment_tracking import ModelSelection, MLFlowTracker
from src.train import Training


# from model import RandomForest,XGBoost
# from data import Preprocessing
# from experiment_tracking import ModelSelection, MLFlowTracker
# from train import Training
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf



from config_dir.config_struct import ChurnConfig


import pandas as pd
cs = ConfigStore.instance()
cs.store(name="churn_config", node=ChurnConfig)
warnings.filterwarnings('ignore')

@hydra.main(config_path='config_dir', config_name='config')
def main(cfg):
    
    # cfg = OmegaConf.load('./config_dir/config.yaml')
    # print(OmegaConf.to_yaml(cfg),"asasas")
    # tracker = MLFlowTracker(cfg.churn_names.experiment_name,cfg.churn_paths.mlflow_tracki)
    tracker = MLFlowTracker(cfg.churn_names.experiment_name,cfg.churn_paths.mlflow_tracking_uri)
    # print(data_path)
    new_data_read = pd.read_csv(cfg.churn_paths.train_dir)
    # previous_data_read=  pd.read_csv('E:/FYP/mlops/MLOps-FYP/dataset/train/Churn_Modelling.csv')
    # previous_data_read=  pd.read_csv(cfg.churn_paths.train_dir)

    
    #Data Merger
    # merged_data = pd.concat([previous_data_read,new_data_read],axis='rows')
    # merged_data.to_csv('E:/FYP/mlops/MLOps-FYP/dataset/train/files/merged.csv')  
    
    preprocessing = Preprocessing()
    pre_processed_dataset = preprocessing.preprocess(
        0.2,new_data_read)
    

    #  pre_processed_dataset = preprocessing.preprocess(
    #     cfg.churn_params.train_test_split,merged_data)


    # Random Forest Model Creation
    model= RandomForest()
    # model = RandomForest(cfg.params.n_estimators, cfg.params.criterion)
    model = model.create_model()
    # tracker.log_params(cfg.params.n_estimators,cfg.params.criterion)
    best_selected_model = Training(
        model, pre_processed_dataset, tracker, cfg.churn_names.metric_name,cfg.churn_paths.fig_path)
    best_selected_model.train()

    print("Model Trained Successfully !!!!!")







if __name__ == "__main__":
    main()
