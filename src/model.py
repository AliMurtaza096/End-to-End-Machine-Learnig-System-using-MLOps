import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


class Models(ABC):
    """
    Abstract base class that defines and creates model.
    """
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass




@dataclass
class RandomForest(Models):
   

    def define_model(self):
        model_parameters = {"n_estimators": [num for num in range(100, 1000, 50)],
                            "criterion": ['gini', 'entropy'],
                            "max_depth": [num for num in range(1, 11)],
                            'bootstrap': [True, False]
                            }
        model_parameters['max_depth'].append(None)
        classifier = RandomForestClassifier()
        tuner = RandomizedSearchCV(
            estimator=classifier, param_distributions=model_parameters, random_state=3, n_jobs=-1)
        return tuner

    def create_model(self):
        model = self.define_model()

        return model


@dataclass
class XGBoost(Models):

    def define_model(self):
        classifier = GradientBoostingClassifier()
        model_parameters = {"n_estimators": [num for num in range(100, 1000, 50)],
                            "criterion": ['friedman_mse', 'squared_error'],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [num for num in range(1, 11)]
                            }

        model_parameters['max_depth'].append(None)
        tuner = RandomizedSearchCV(
            estimator=classifier, param_distributions=model_parameters, random_state=3, n_jobs=-1)

        return tuner

    def create_model(self):
        model = self.define_model()

        return model
