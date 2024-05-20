
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from apps.core.logger import Logger
from sklearn.metrics import r2_score


class ModelTuner:
    """Model Tuner class

    This class performs hyperparameter tuning for machine learning models.

    Attributes:
        run_id (str): Unique identifier for the tuning run.
        data_path (str): Path to the dataset file.
        logger (obj): Logger object for logging messages.
        rfc (sklearn.ensemble.RandomForestClassifier): Random Forest classifier object (default).
        xgb (xgboost.XGBClassifier): XGBoost classifier object (default for binary classification).

    """
    def __init__(self, run_id, data_path, mode):
        self.random_forest_score = None
        self.prediction_random_forest = None
        self.random_forest = None
        self.xgboost_score = None
        self.prediction_xgboost = None
        self.xgboost = None
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'ModelTuner', mode)
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')  # For binary classification

    def best_params_randomforest(self, train_x, train_y):
        """Finds the best hyperparameters for the Random Forest classifier.

        This method uses GridSearchCV to perform hyperparameter tuning for the
        Random Forest classifier. It evaluates different combinations of parameters
        and selects the set that yields the best accuracy on the training data.

        Args:
            train_x (numpy.ndarray): The training data features.
            train_y (numpy.ndarray): The training data target labels.

        Returns:
            sklearn.ensemble.RandomForestClassifier: The Random Forest classifier
                trained with the best hyperparameters.

        Raises:
            Exception: If an error occurs during hyperparameter tuning.
        """
        try:
            self.logger.info('Start of finding best params for Random Forest...')

            # Define a grid of hyperparameter values to explore
            param_grid = {
                "n_estimators": [10, 50, 100, 130],
                "criterion": ['gini', 'entropy'],
                "max_depth": range(2, 4),
                "max_features": ['auto', 'log2']
            }

            # Create a GridSearchCV object
            grid = GridSearchCV(estimator=self.rfc, param_grid=param_grid, cv=5)

            # Perform grid search
            grid.fit(train_x, train_y)

            # Extract the best parameters
            self.rfc.n_estimators = grid.best_params_['n_estimators']
            self.rfc.criterion = grid.best_params_['criterion']
            self.rfc.max_depth = grid.best_params_['max_depth']
            self.rfc.max_features = grid.best_params_['max_features']

            # Create a new Random Forest model with the best parameters
            self.rfc = RandomForestClassifier(**grid.best_params_)

            # Train the model with the best parameters
            self.rfc.fit(train_x, train_y)

            self.logger.info('Random Forest best params: {}'.format(grid.best_params_))
            self.logger.info('End of finding best params for Random Forest...')
            return self.rfc

        except Exception as e:
            self.logger.exception('Exception raised while finding best params for Random Forest: {}'.format(e))
            raise Exception()

    def best_params_xgboost(self, train_x, train_y):
        """Finds the best hyperparameters for the XGBoost classifier.

        This method uses GridSearchCV to perform hyperparameter tuning for the
        XGBoost classifier. It evaluates different combinations of parameters
        and selects the set that yields the best accuracy on the training data.

        Args:
            train_x (numpy.ndarray): The training data features.
            train_y (numpy.ndarray): The training data target labels.

        Returns:
            xgboost.XGBClassifier: The XGBoost classifier trained with the best hyperparameters.

        Raises:
            Exception: If an error occurs during hyperparameter tuning.
        """

        try:
            self.logger.info('Start of finding best params for XGBoost...')

            # Define a grid of hyperparameter values to explore
            param_grid = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]
            }

            # Create a GridSearchCV object
            grid = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic'), param_grid=param_grid, cv=5)

            # Perform grid search
            grid.fit(train_x, train_y)

            # Extract the best parameters
            self.xgb.learning_rate = grid.best_params_['learning_rate']
            self.xgb.max_depth = grid.best_params_['max_depth']
            self.xgb.n_estimators = grid.best_params_['n_estimators']

            # Train the XGBoost model with the best parameters
            self.xgb.fit(train_x, train_y)

            self.logger.info('XGBoost best params: {}'.format(grid.best_params_))
            self.logger.info('End of finding best params for XGBoost algo...')
            return self.xgb

        except Exception as e:
            self.logger.exception('Exception raised while finding best params for XGBoost: {}'.format(e))
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """Selects the best performing model between XGBoost and Random Forest.

        This method trains both XGBoost and Random Forest models using hyperparameter
        tuning on the training data. It then evaluates their performance on the test
        data using either accuracy (for single-class problems) or AUC-ROC score
        (for multi-class problems). The model with the higher score is selected
        as the best model.

        Args:
            train_x (numpy.ndarray): The training data features.
            train_y (numpy.ndarray): The training data target labels.
            test_x (numpy.ndarray): The test data features.
            test_y (numpy.ndarray): The test data target labels.

        Returns:
            tuple: A tuple containing the name of the best model ('XGBoost' or 'RandomForest')
                and the corresponding trained model object.

        Raises:
            Exception: If an error occurs during model training or evaluation.
        """

        try:
            self.logger.info('Start of finding best model...')

            # Train and evaluate XGBoost model
            self.xgboost = self.best_params_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)

            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger.info('Accuracy for XGBoost: {}'.format(self.xgboost_score))
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger.info('AUC for XGBoost: {}'.format(self.xgboost_score))

            # Train and evaluate Random Forest model
            self.random_forest = self.best_params_randomforest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)

            if len(test_y.unique()) == 1:
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger.info('Accuracy for Random Forest: {}'.format(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)
                self.logger.info('AUC for Random Forest: {}'.format(self.random_forest_score))

            # Identify the best model
            best_model = 'XGBoost' if self.xgboost_score > self.random_forest_score else 'RandomForest'
            best_model_object = self.xgboost if best_model == 'XGBoost' else self.random_forest

            self.logger.info('End of finding best model...')
            return best_model, best_model_object

        except Exception as e:
            self.logger.exception('Exception raised while finding best model: {}'.format(e))
            raise Exception()
