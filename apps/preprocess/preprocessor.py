import json
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from apps.core.logger import Logger


class Preprocessor:
    """
    A class for preprocessing training and prediction datasets.

    Attributes:
        run_id (str): The unique identifier for the current run.
        data_path (str): The path to the dataset.
        logger (Logger): The logger instance for logging.
    """

    def __init__(self, run_id, data_path, mode, target_column):
        """
        Initialize the Preprocessor class.

        Args:
            run_id (str): The unique identifier for the current run.
            data_path (str): The path to the dataset.
            mode (str): The mode of the logger (e.g., 'train', 'predict').
            target_column (str): Specifies the target column name

        Returns:
            None
        """
        self.y = None
        self.X = None
        self.null_present = None
        self.null_counts = None
        self.columns = None
        self.useful_data = None
        self.data = None
        self.target_column = target_column
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'Preprocessor', mode)

    def get_data(self):
        """
        Method to read data from a file.

        Returns:
            pandas.DataFrame: The read DataFrame.

        Raises:
            Exception: If an error occurs while reading the dataset.
        """
        try:
            self.logger.info('Start of reading dataset...')
            self.data = pd.read_csv(self.data_path + '_validation/InputFile.csv')
            self.logger.info(f"Shape of the just read data: {self.data.shape}")
            self.logger.info(f"End of reading dataset from {self.data_path + '_validation/InputFile.csv'}")
            return self.data
        except Exception as e:
            self.logger.exception('Exception raised while reading dataset: %s' % e)
            raise Exception()

    def drop_columns(self, data, columns):
        """
        Method to drop specified columns from a pandas DataFrame.

        Args:
            data (pandas.DataFrame): The input DataFrame.
            columns (list): A list of column names to be dropped.

        Returns:
            pandas.DataFrame: A DataFrame after removing the specified columns.

        Raises:
            Exception: If an error occurs during column dropping.
        """
        self.data = data
        self.columns = columns
        try:
            self.logger.info('Start of Dropping Columns...')
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger.info('End of Dropping Columns...')
            return self.useful_data
        except Exception as e:
            self.logger.exception('Exception raised while Dropping Columns: %s' % e)
            raise Exception()

    def is_null_present(self, data):
        """
        Method to check if null values are present in the DataFrame.

        Args:
            data (pandas.DataFrame): The input DataFrame.

        Returns:
            bool: True if null values are present, False if they are not present.

        Raises:
            Exception: If an error occurs while finding missing values.
        """
        self.null_present = False
        try:
            self.logger.info('Start of finding missing values...')
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present:
                # Create a DataFrame with column names and missing values count
                dataframe_with_null = pd.DataFrame({'columns': data.columns,
                                                    'missing values count': np.asarray(data.isna().sum())})
                # Store the null column information to file
                dataframe_with_null.to_csv(self.data_path + '_validation/' + 'null_values.csv')
            self.logger.info('End of finding missing values...')
            return self.null_present
        except Exception as e:
            self.logger.exception('Exception raised while finding missing values: %s' % e)
            raise Exception()

    def impute_missing_values(self, data):
        """
        Method to impute missing values using KNNImputer.

        Args:
            data (pandas.DataFrame): The input DataFrame containing missing values.

        Returns:
            pandas.DataFrame: A DataFrame with missing values imputed.

        Raises:
            Exception: If an error occurs during imputation.
        """
        try:
            self.logger.info('Start of imputing missing values...')
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            new_array = imputer.fit_transform(data)
            new_data = pd.DataFrame(data=new_array, columns=data.columns)
            self.logger.info('End of imputing missing values...')
            return new_data
        except Exception as e:
            self.logger.exception('Exception raised while imputing missing values: %s' % e)
            raise Exception()

    def feature_encoding(self, data):
        """
        Feature encodes categorical variables using dummy encoding.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data containing categorical variables.

        Returns
        -------
        pandas.DataFrame
            DataFrame with categorical variables encoded.
        """
        try:
            self.logger.info('Start of feature encoding...')

            # Selecting only object type columns
            categorical_data = data.select_dtypes(include=['object']).copy()

            # Encoding categorical columns
            encoded_data = pd.get_dummies(categorical_data, drop_first=True)

            self.logger.info('End of feature encoding...')
            return encoded_data
        except Exception as e:
            self.logger.exception('Exception raised while feature encoding: ' + str(e))
            raise

    def split_features_label(self, data, label_name):
        """
        Split features and label from the input data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data containing both features and labels.
        label_name : str
            Name of the column containing the label.

        Returns
        -------
        tuple
            A tuple containing two elements:
                - X: DataFrame containing the features.
                - y: Series containing the labels.

        Raises
        ------
        Exception
            If any error occurs during the process.
        """
        self.data = data
        try:
            self.logger.info('Start of splitting features and label...')
            self.X = self.data.drop(labels=label_name, axis=1)  # drop the label column to separate the features
            self.y = self.data[label_name]  # Extract the label column
            self.logger.info('End of splitting features and label...')
            return self.X, self.y
        except Exception as e:
            self.logger.exception('Exception raised while splitting features and label: ' + str(e))
            raise

    def final_predict_set(self, data):
        """
        Build the final prediction set by adding another encoded column with a default value of 0.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data to be included in the prediction set.

        Returns
        -------
        pandas.DataFrame
            The final prediction set with added columns and filled missing values with 0.

        Raises
        ------
        ValueError
            If a ValueError occurs while building the final prediction set.
        KeyError
            If a KeyError occurs while building the final prediction set.
        Exception
            If any other exception occurs while building the final prediction set.

        """
        try:
            self.logger.info('Start of building final prediction set...')
            with open('apps/database/columns.json', 'r') as f:
                data_columns = json.load(f)['data_columns']
            df = pd.DataFrame(data=None, columns=data_columns)
            df_new = pd.concat([df, data], ignore_index=True, sort=False)
            data_new = df_new.fillna(0)
            self.logger.info('End of building final prediction set...')
            return data_new
        except ValueError:
            self.logger.exception('ValueError raised while building final prediction set')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while building final prediction set')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while building final prediction set: %s' % e)
            raise e

    def preprocess_train_set(self):
        """
        Preprocesses the training data.

        Returns
        -------
        tuple
            A tuple containing two elements:
                - X: DataFrame containing the features.
                - y: Series containing the labels.

        Raises
        ------
        Exception
            If any error occurs during the preprocessing.

        """
        try:
            self.logger.info('Start of Preprocessing...')
            # get data into pandas DataFrame
            data = self.get_data()
            # drop unwanted columns
            data = self.drop_columns(data, ['empid'])
            # handle label encoding
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, [self.target_column])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation
            # create separate features and labels
            self.X, self.y = self.split_features_label(data, label_name='left')
            self.logger.info('End of Preprocessing...')
            return self.X, self.y
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predict_set(self):
        """
        Preprocesses the prediction data.

        Returns
        -------
        pandas.DataFrame
            Preprocessed prediction data.

        Raises
        ------
        Exception
            If any error occurs during the preprocessing.

        """
        try:
            self.logger.info('Start of Preprocessing...')
            # get data into pandas DataFrame
            data = self.get_data()
            # handle label encoding
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)

            data_shape = data.shape

            self.logger.info(f"This is data shape: {data_shape}")

            # drop categorical column
            data = self.drop_columns(data, [self.target_column])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation

            data = self.final_predict_set(data)
            self.logger.info('End of Preprocessing...')
            return data
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predict(self, data):
        """
        Preprocesses the prediction data.

        Parameters
        ----------
        data : pandas.DataFrame
            The prediction data to be preprocessed.

        Returns
        -------
        pandas.DataFrame
            Preprocessed prediction data.

        Raises
        ------
        Exception
            If any error occurs during the preprocessing.

        """
        try:
            self.logger.info('Start of Preprocessing...')
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, [self.target_column])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation

            data = self.final_predict_set(data)
            self.logger.info('End of Preprocessing...')
            return data
        except Exception as e:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            print(e)
