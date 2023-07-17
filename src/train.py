from src.data import Dataset
from src.model import RandomForest
from src.experiment_tracking import MLFlowTracker, ModelSelection


# from data import Dataset
# from model import RandomForest
# from experiment_tracking import MLFlowTracker, ModelSelection

from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Training:
    def __init__(self, model, data: Dataset, tracker: MLFlowTracker, metric_name: str,
                 fig_path:str) -> None:
        self.model = model
        self.data = data
        self.tracker = tracker
        self.metric_name = metric_name
        self.fig_path= fig_path

    # Training the model and the logging the artifacts using mlflow
    # MLflow tracker is named as self.tracker

    def train(self):

        classifier = self.model.fit(self.data.x_train, self.data.y_train)
        best_params = classifier.best_params_
        y_pred = classifier.predict(self.data.x_test)
        # For Random Forest
        self.tracker.log_params(n_estimators=best_params['n_estimators'], criterion=best_params['criterion'],
                                max_depth=best_params['max_depth'], bootstrap=best_params['bootstrap'])

        # Model for Gradient Boosting
        # self.tracker.log_params(n_estimators=best_params['n_estimators'],criterion=best_params['criterion'],
        #                         max_depth=best_params['max_depth'],learning_rate=best_params['learning_rate'])

        self.tracker.log_model(classifier, self.data.x_train,
                               classifier.predict(self.data.x_train))
        accuracy = accuracy_score(self.data.y_test, y_pred)
        precision = precision_score(self.data.y_test, y_pred)
        recall = recall_score(self.data.y_test, y_pred)
        
        self.tracker.log_metrics(accuracy, precision, recall)

        
        #Confusion Matrix
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            self.data.y_test,
            y_pred,
            ax=ax,
            colorbar=False)
        # save the plot
        plt.savefig(self.fig_path)
        self.tracker.log_figure(fig, 'confusion_matrix.png')

        return ModelSelection(self.tracker.find_best_model(self.metric_name))
