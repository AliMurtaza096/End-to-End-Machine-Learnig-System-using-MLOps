from dataclasses import dataclass


@dataclass
class ChurnNames:
    experiment_name: str
    metric_name: str


@dataclass
class ChurnPaths:
    train_dir: str
    mlflow_tracking_uri: str
    model_artifactory_dir: str
    fig_path: str

@dataclass
class ChurnParams:
    train_test_split: float
    


@dataclass
class ChurnConfig:
    churn_paths: ChurnPaths
    churn_params: ChurnParams
    churn_names: ChurnNames


@dataclass
class ImageNames:
    experiment_name: str
    metric_name: str
    
    
@dataclass
class ImagePaths:
    train_dir: str
    mlflow_tracking_uri: str
    model_artifactory_dir: str


@dataclass
class ImageParams:
    train_test_split: float
    batch_size:int 
    epochs:int
    lr: float
    
    

@dataclass
class ImageConfig:
    paths: ImagePaths
    params: ImageParams
    names: ImageNames
    
