from apps.core.logger import Logger
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from apps.core.file_operation import FileOperation
from apps.tuning.model_tuner import ModelTuner
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor


class KMeansCluster:
    """KMeans Cluster class

    This class implements a KMeans clustering algorithm.

    Attributes:
        run_id (str): Unique identifier for the clustering run.
        data_path (str): Path to the dataset file.
        logger (obj): Logger object for logging messages.
        fileOperation (obj): File operation object for data manipulation.
    """

    def __init__(self, run_id, data_path):
        self.saveModel = None
        self.y_kmeans = None
        self.kmeans = None
        self.data = None
        self.kn = None
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'KMeansCluster', 'training')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'training')

    def elbow_plot(self, data):
        """Calculates the optimal number of clusters using the Elbow Method.

        This method plots the WCSS (Within Cluster Sum of Squares) for a range of
        cluster numbers and identifies the elbow point, which indicates the optimal
        number of clusters for the data.

        Args:
            data (numpy.ndarray): The dataset to be clustered.

        Returns:
            int: The optimal number of clusters identified by the Elbow Method.

        Raises:
            Exception: If an error occurs during elbow plot generation.
        """

        wcss = []  # Initialize an empty list to store WCSS values

        try:
            self.logger.info('Start of elbow plotting...')
            for num_clusters in range(1, 11):
                kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)

            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('apps/models/kmeans_elbow.png')  # Save the elbow plot

            # Programmatically find the optimal number of clusters using KneeLocator
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger.info('The optimum number of clusters is: ' + str(self.kn.knee))
            self.logger.info('End of elbow plotting...')
            return self.kn.knee

        except Exception as e:
            self.logger.exception('Exception raised while elbow plotting:' + str(e))
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """Creates clusters for the provided data using the KMeans algorithm.

        This method performs KMeans clustering on the data and assigns each data point
        to a cluster. It then adds a new column named "Cluster" to the data frame
        containing the cluster labels for each data point.

        Args:
            data (numpy.ndarray): The dataset to be clustered.
            number_of_clusters (int): The desired number of clusters.

        Returns:
            pandas.DataFrame: The original data frame with an added "Cluster" column
                containing cluster labels for each data point.

        Raises:
            Exception: If an error occurs during cluster creation.
        """

        self.data = data
        try:
            self.logger.info('Start of creating clusters...')
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=0)
            self.y_kmeans = self.kmeans.fit_predict(data)  # Fit and predict cluster labels

            # Saving the KMeans model (assuming 'fileOperation' handles model saving)
            self.saveModel = self.fileOperation.save_model(self.kmeans, 'KMeans')

            self.data['Cluster'] = self.y_kmeans  # Add cluster labels as a new column

            # Assuming 'kn.knee' is already calculated by 'elbow_plot'
            self.logger.info(f'Successfully created {number_of_clusters} clusters.')
            self.logger.info('End of creating clusters...')
            return self.data

        except Exception as e:
            self.logger.exception(f'Exception raised while creating clusters: {e}')
            raise Exception()
