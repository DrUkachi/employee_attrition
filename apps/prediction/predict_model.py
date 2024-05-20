import pandas as pd
from apps.core.logger import Logger
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.core.file_operation import FileOperation


class PredictModel:

    """
    Class for performing predictions using a trained model.

    This class encapsulates the logic for loading, validating, preprocessing,
    and making predictions using a trained model. It takes the run ID and data path
    as arguments during initialization and provides methods for:

    - `batch_predict_from_model`: Performs batch prediction on the provided data path.
    - `single_predict_from_model`: Performs a single prediction on provided data.

    Additionally, the class utilizes several helper objects:

    - `Logger`: Handles logging for the prediction process.
    - `LoadValidate`: Responsible for loading and validating the prediction data.
    - `Preprocessor`: Handles preprocessing steps on the prediction data.
    - `FileOperation`: Provides functionalities for file operations related to prediction.

    """

    def __init__(self, run_id, data_path):
        self.X = None
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'PredictModel', 'prediction')
        self.loadValidate = LoadValidate(self.run_id, self.data_path, 'prediction')
        self.preProcess = Preprocessor(self.run_id, self.data_path, 'prediction', target_column='salary')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'prediction')

    def batch_predict_from_model(self):
        """Performs batch prediction on the data specified by the data path.

            This method performs the following steps for batch prediction:

            1. **Validation and Transformation:**
                - Loads and validates the prediction data using `self.loadValidate.validate_predict_set()`.

            2. **Preprocessing:**
                - Preprocesses the loaded data using `self.preProcess.preprocess_predict_set()`.

            3. **Model Loading and Prediction:**
                - Loads the KMeans model using `self.fileOperation.load_model('KMeans')`.
                - Predicts clusters for each data point using the KMeans model.
                - Iterates through unique clusters:
                - Filters data belonging to the current cluster.
                - Loads the appropriate model based on the cluster using `self.fileOperation.correct_model(i)`.
                - Performs prediction on the filtered data using the loaded model.
                - Creates a DataFrame with predicted results for the current cluster and saves it to a CSV file.

            4. **Logging:**
                - Logs the start and end of the prediction process.

            **Raises:**

                - Exception: If any error occurs during the prediction process.

        """
        try:
            self.logger.info('Start of Prediction')
            self.logger.info('run_id:' + str(self.run_id))
            # validations and transformation
            self.loadValidate.validate_predict_set()
            # preprocessing activities
            self.X = self.preProcess.preprocess_predict_set()

            shape_value = self.X.shape

            self.logger.info(f"This is the shape of the batch prediction data {shape_value}")
            # load model
            kmeans = self.fileOperation.load_model('KMeans')
            # cluster selection
            clusters = kmeans.predict(self.X.drop(['empid'], axis=1))
            self.X['clusters'] = clusters
            clusters = self.X['clusters'].unique()
            y_predicted = []
            for i in clusters:
                self.logger.info('clusters loop started')
                cluster_data = self.X[self.X['clusters'] == i]
                cluster_data_new = cluster_data.drop(['empid','clusters'], axis=1)
                model_name = self.fileOperation.correct_model(i)
                model = self.fileOperation.load_model(model_name)
                y_predicted = model.predict(cluster_data_new)
                # result = pd.DataFrame(list(zip(y_predicted)), columns=['Predictions'])
                # result.to_csv(self.data_path+'_results/'+'Predictions.csv', header=True, mode='a+')
                result = pd.DataFrame({"EmpId": cluster_data['empid'], "Prediction": y_predicted})
                result.to_csv(self.data_path+'_results/'+'Predictions.csv', header=True, mode='a+',index=False)
            self.logger.info('End of Prediction')
        except Exception:
            self.logger.exception('Unsuccessful End of Prediction')
            raise Exception

    def single_predict_from_model(self, data):
        """
        Performs a single prediction using the trained model.

        This method takes a DataFrame `data` containing the features for a single prediction.
        It preprocesses the data (if necessary) and makes a prediction using the trained model.
        The predicted output is returned.

        Args:
        data (pandas.DataFrame): A DataFrame containing the features for a single prediction.

        Returns:
            Any: The predicted output from the model. The data type depends on the model's output.
        """
        try:
            self.logger.info('Start of Prediction')
            self.logger.info('run_id:' + str(self.run_id))
            # preprocessing activities
            self.X = self.preProcess.preprocess_predict(data)
            # load model
            kmeans = self.fileOperation.load_model('KMeans')
            # cluster selection
            clusters = kmeans.predict(self.X.drop(['empid'], axis=1))
            self.X['clusters'] = clusters
            clusters = self.X['clusters'].unique()
            y_predicted = []
            for i in clusters:
                self.logger.info('clusters loop started')
                cluster_data = self.X[self.X['clusters'] == i]
                cluster_data_new = cluster_data.drop(['empid', 'clusters'], axis=1)
                model_name = self.fileOperation.correct_model(i)
                model = self.fileOperation.load_model(model_name)
                self.logger.info('Shape of Data '+str(cluster_data_new.shape))
                self.logger.info('Shape of Data ' + str(cluster_data_new.info()))
                y_predicted = model.predict(cluster_data_new)
                # result = pd.DataFrame(list(zip(y_predicted)), columns=['Predictions'])
                # result.to_csv(self.data_path+'_results/'+'Predictions.csv', header=True, mode='a+')
                # result = pd.DataFrame({"EmpId": cluster_data['empid'],"Prediction": y_predicted})
                # result.to_csv(self.data_path+'_results/'+'Predictions.csv', header=True, mode='a+',index=False)
                self.logger.info('Output : '+str(y_predicted))
                self.logger.info('End of Prediction')
                return int(y_predicted[0])
        except Exception as e:
            self.logger.exception('Unsuccessful End of Prediction')
            raise e