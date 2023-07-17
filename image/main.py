import warnings
import hydra

from data import Dataset,Preprocessing
from model import EfficientNetB3
from experiment_tracking import MLFlowTracker, ModelSelection
from train import Training


from hydra.core.config_store import ConfigStore

from config_dir.config_struct import ImageConfig
cs = ConfigStore.instance()
cs.store(name="image_config", node=ImageConfig)
warnings.filterwarnings('ignore')

@hydra.main(config_path='config_dir', config_name='config')
def main(cfg):
    tracker = MLFlowTracker(cfg.image_names.experiment_name,cfg.image_paths.mlflow_tracking_uri)
    tracker.log()

    dataset = Dataset()
    preprocess = Preprocessing(cfg.image_paths.train_dir,cfg.image_params.train_test_split)
    processed_data = preprocess.data_augmentation()
    
    model = EfficientNetB3(processed_data.train_gen).create_model()

    print(model)
    best_model  = Training(model=model,data=processed_data,tracker=tracker,
                           metric_name=cfg.image_names.metric_name,batch_size=cfg.image_params.batch_size,
                           epochs=cfg.image_params.epochs,learning_rate=cfg.image_params.lr).train()
    


if __name__ == "__main__":
    main()