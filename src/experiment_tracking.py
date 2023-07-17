from abc import ABC, abstractmethod
import mlflow
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
    def log_metrics(self):
        pass

    @abstractmethod
    def log_model(self):
        pass

    @abstractmethod
    def log_params(self):
        pass

    @abstractmethod
    def find_best_model(self):
        pass


@dataclass
class ModelSelection:
    model_selection_dataframe: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame())


@dataclass
class MLFlowTracker:

    experiment_name: str
    tracking_uri: str

    with mlflow.start_run(run_name='Churn Prediction') as run:
        def __start__(self):
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

        def log_params(self, **kwargs):
            self.__start__()
            params = kwargs
            mlflow.log_params(params)

        def log_metrics(self, accuracy, precision, recall):
            self.__start__()

            metrics = {'accuracy': accuracy,
                       'precision': precision, 'recall': recall}
            mlflow.log_metrics(metrics)

        def log_model(self, model, model_input, model_output):
            self.__start__()

            signature = infer_signature(model_input, model_output)
            mlflow.sklearn.log_model(
                model, "model", registered_model_name='Bank Churn Prediction', signature=signature)

        def log_figure(self, fig, path):
            self.__start__()
            mlflow.log_figure(fig, path)

        def find_best_model(self, metric):

            experiment = dict(
                mlflow.get_experiment_by_name(self.experiment_name))
            experiment_id = experiment['experiment_id']

            result_df = mlflow.search_runs([experiment_id],
                                           order_by=[f"metrics.{metric} DESC"])
            # print(result_df["experiment_id", "run_id", f"metrics.{metric}"][0])
            return ModelSelection(model_selection_dataframe=result_df[
                ["experiment_id", "run_id", f"metrics.{metric}"]
            ])
