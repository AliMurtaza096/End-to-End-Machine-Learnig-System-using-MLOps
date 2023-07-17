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


@dataclass
class ChurnParams:
    n_estimators: int
    criterion: str
    train_test_split: str


@dataclass
class ChurnConfig:
    churn_paths: ChurnPaths
    churn_params: ChurnParams
    churn_names: ChurnNames

@dataclass
class ImagePaths:
    train_dir: str
    mlflow_tracking_uri: str
    model_artifactory_dir: str


@dataclass
class ImageParams:
    n_estimators: int
    criterion: str
    train_test_split: str


@dataclass
class ImageNames:
    experiment_name: str
    metric_name: str


@dataclass
class ImageConfig:
    paths: ImagePaths
    params: ImageParams
    names: ImageNames
    
