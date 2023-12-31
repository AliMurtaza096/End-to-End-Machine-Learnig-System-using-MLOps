from abc import ABC, abstractmethod
import mlflow
import mlflow.keras as mk
import mlflow.tensorflow as mt
import pandas as pd
from dataclasses import dataclass, field
from mlflow.models.signature import infer_signature

class ExperimentTracking(ABC):
    """
    Interface to track experiments by inherting from Protocol class.
    """

    def __start__(self):
        pass
    
    @abstractmethod
    def log(self):
        pass
    

    @abstractmethod
    def find_best_model(self):
        pass


@dataclass
class ModelSelection:
    """
    Dataclass that contains the dataframe with sorted list of models based on the
    given metric.
    Instance variables
    ------------------
    model_selection_dataframe: DataFrame
    """

    model_selection_dataframe: pd.DataFrame = field(default_factory = lambda: pd.DataFrame())

@dataclass
class MLFlowTracker:
    """
    Dataclass to track experiment via MLFlow.
    Instance variables
    ------------------
    experiment_name: str
        Name of the experiment to be activated.
    tracking_uri: str
        An HTTP URI or local file path, prefixed with `file:/`
   
    Returns
    -------
        None.
    """
    experiment_name: str
    tracking_uri: str
    
    with mlflow.start_run(run_name='') as run:
    
        def __start__(self) -> None:
            """
            Dunder method that sets tracking URI and experiment name
            to MLFlow engine.
            """
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        def log_params(self, **kwargs):
            self.__start__()
            params = kwargs
            mlflow.log_params(params)
        def log(self) -> None:
            """
            Initialize auto-logging for tracking. This will log model
            artifacts, parameters and metrics in the ./artifacts directory.
            """
            self.__start__()
            mlflow.start_run
            mk.autolog(log_models=True ,registered_model_name='X-Ray Image Disease Classification')

        def log_model(self, model):
                self.__start__()

                # signature = infer_signature(model_input, model_output)
                mk.log_model(model,"model",registered_model_name='X-Ray Image Disease Classification')
                
        def log_figure(self, fig, path):
                self.__start__()
                mlflow.log_figure(fig, path)
        def save_model(self,model,path):
            self.__start__()
            mk.save_model(model,path)
             
        def find_best_model(self, metric: str) -> ModelSelection(pd.DataFrame):
            """
            Method for model selection. Provides functionalities to find and sort
            the best model based on the given metric in descending order from all
            models within the given experiment directory which makes it easier to
            select best performing model.
            Note: This can also be done with mlflow using `mlflow ui` command. But,
            this is a code implementation of the same.
            Parameters
            ----------
            metric: str
                Metric name to sort the models.
            Returns
            -------
            instanceof: ModelSelection(pd.DataFrame)
                Resulting dataframe.
            Raises
            ------
            MLFlowError: Exception
                If the experiment id or experiment name is none/invalid.
            """
            
            experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            experiment_id = experiment['experiment_id']

            # if experiment is None or experiment_id is None:
            #     raise MLFlowError(
            #         f"Invalid experiment details. Please re-check them and try again !!!")

            result_df = mlflow.search_runs([experiment_id], 
                                            order_by=[f"metrics.{metric} DESC"])
            print(result_df,"aasa")
            return ModelSelection(model_selection_dataframe = result_df[
                                            ["experiment_id", "run_id", f"metrics.{metric}"]
                                            ])