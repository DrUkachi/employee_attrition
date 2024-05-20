from apps.core.logger import Logger
import json
from sklearn.model_selection import train_test_split
from apps.core.file_operation import FileOperation
from apps.tuning.model_tuner import ModelTuner
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.tuning.cluster import KMeansCluster


class TrainModel:
    """
    Class for training models.

    Attributes
    ----------
    run_id : str
        The ID associated with the training run.
    data_path : str
        The path to the training data.
    logger : Logger
        Logger instance for logging.
    load_validate : LoadValidate
        Instance of LoadValidate for data loading and validation.
    preprocessor : Preprocessor
        Instance of Preprocessor for data preprocessing.
    model_tuner : ModelTuner
        Instance of ModelTuner for model tuning.
    file_operation : FileOperation
        Instance of FileOperation for file operations.
    cluster : KMeansCluster
        Instance of KMeansCluster for clustering.

    Methods
    -------
    __init__(run_id, data_path)
        Initializes TrainModel with run_id and data_path.
    training_model()
        Trains the model.
    """

    def __init__(self, run_id, data_path):
        """
        Initializes TrainModel with run_id and data_path.

        Parameters
        ----------
        run_id : str
            The ID associated with the training run.
        data_path : str
            The path to the training data.
        """
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'TrainModel', 'training')
        self.load_validate = LoadValidate(self.run_id, self.data_path, 'training')
        self.preprocessor = Preprocessor(self.run_id, self.data_path,
                                         mode='training', target_column="salary")
        self.model_tuner = ModelTuner(self.run_id, self.data_path, 'training')
        self.file_operation = FileOperation(self.run_id, self.data_path, 'training')
        self.cluster = KMeansCluster(self.run_id, self.data_path)

        self.X = None
        self.y = None

    def training_model(self):
        """
        Trains the model.

        Raises
        ------
        Exception
            If any error occurs during the training process.
        """
        try:
            self.logger.info('Start of Training')
            self.logger.info('Run_id:' + str(self.run_id))

            # Load, validate, and transform the training set
            self.load_validate.validate_train_set()

            # Preprocess the training set
            self.X, self.y = self.preprocessor.preprocess_train_set()

            # Save column names to a JSON file
            columns = {"data_columns": [col for col in self.X.columns]}
            with open('apps/database/columns.json', 'w') as f:
                f.write(json.dumps(columns))

            # Determine the number of clusters
            number_of_clusters = self.cluster.elbow_plot(self.X)

            # Divide the data into clusters
            self.X = self.cluster.create_clusters(self.X, number_of_clusters)

            # Add a new column to the dataset consisting of cluster assignments
            self.X['Labels'] = self.y

            # Get unique clusters from the dataset
            list_of_clusters = self.X['Cluster'].unique()

            # Iterate over each cluster
            for i in list_of_clusters:
                cluster_data = self.X[self.X['Cluster'] == i]  # Filter the data for one cluster

                # Prepare feature and label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # Split data into training and test sets for each cluster
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.2,
                                                                    random_state=0)

                # Get the best model for each cluster
                best_model_name, best_model = self.model_tuner.get_best_model(x_train, y_train, x_test, y_test)

                # Save the best model to the directory
                save_model = self.file_operation.save_model(best_model, best_model_name + str(i))
            self.logger.info('End of Training')
        except Exception:
            self.logger.exception('Unsuccessful End of Training')
            raise Exception
