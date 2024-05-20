
import pickle
import os
import shutil
from apps.core.logger import Logger


class FileOperation:
    """File operation helper class.

    This class provides helper methods for file operations used during
    model training and saving.

    Attributes:
        run_id (str): Unique identifier for the training run.
        data_path (str): Path to the dataset file.
        logger (obj): Logger object for logging messages.
    """

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'FileOperation', mode)

        # Avoid initializing instance variables with potentially unused values
        self.model_name = None
        self.file = None
        self.list_of_files = None
        self.list_of_model_files = None
        self.folder_name = None
        self.cluster_number = None

    def save_model(self, model, model_name):
        """Saves a machine learning model to a file.

        This method saves the provided `model` object to a file named
        `model_name.sav` within the `apps/models` directory. It creates the
        directory if it doesn't exist and removes any previously saved models
        with the same name to avoid conflicts.

        Args:
            model: The machine learning model object to be saved.
            model_name (str): The name to be used for the saved model file (without the '.sav' extension).

        Returns:
            str: The string 'success' if the model is saved successfully.

        Raises:
            Exception: If an error occurs during saving the model.
        """

        try:
            self.logger.info('Start of saving model...')

            # Create the directory structure if it doesn't exist
            model_dir = os.path.join('apps', 'models')
            os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

            # Construct the complete file path
            model_path = os.path.join(model_dir, f'{model_name}.sav')

            # Save the model to a file using pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            self.logger.info(f'Model "{model_name}" saved successfully.')
            self.logger.info('End of saving model...')
            return 'success'

        except Exception as e:
            self.logger.exception(f'Exception raised while saving model: {e}')
            raise Exception()

    def load_model(self, file_name):
        """
        Loads the model file.

        Parameters
        ----------
        file_name : str
            The name of the model file to be loaded.

        Returns
        -------
        object
            The loaded model object.
        """
        try:
            self.logger.info('Start of Load Model')
            with open('apps/models/' + file_name + '.sav', 'rb') as f:
                self.logger.info('Model File ' + file_name + ' loaded')
                self.logger.info('End of Load Model')
                return pickle.load(f)
        except Exception as e:
            self.logger.exception('Exception raised while Loading Model: %s' % e)
            raise Exception()

    def correct_model(self, cluster_number):
        """
        Finds the correct model.

        Parameters
        ----------
        cluster_number : int
            The number of the cluster.

        Returns
        -------
        str
            The name of the correct model file.

        """
        try:
            self.logger.info('Start of finding correct model')
            self.cluster_number = cluster_number
            self.folder_name = 'apps/models'
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except Exception as e:
                    print(e)
                    continue

            self.model_name = self.model_name.split('.')[0]
            self.logger.info('End of finding correct model')
            return self.model_name
        except Exception as e:
            self.logger.exception('Exception raised while finding correct model: %s' % e)
            raise Exception()
