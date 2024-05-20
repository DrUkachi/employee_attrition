import os
from os import listdir
import json

import shutil

import pandas as pd
from datetime import datetime

from apps.database.database_operation import DatabaseOperation
from apps.core.logger import Logger


class LoadValidate:
    """
    A class to load, validate, and transform the data.

    Attributes
    ----------
    run_id : str
        The ID for the current run.
    data_path : str
        The path to the data directory.
    logger : Logger
        An instance of the Logger class for logging messages.
    dbOperation : DatabaseOperation
        An instance of the DatabaseOperation class for database operations.

    Methods
    -------
    __init__(run_id, data_path, mode)
        Initializes the LoadValidate instance.

    values_from_schema(self, schema_file)
    Method to read schema file and extract column names and the number of columns.

    validate_column_length(self, number_of_columns)
    Method to validate the number of columns in the CSV files against the provided schema.


    """

    def __init__(self, run_id, data_path, mode):
        """
        Initializes the LoadValidate instance.

        Parameters
        ----------
        run_id : str
            The ID for the current run.
        data_path : str
            The path to the data directory.
        mode : str
            The mode of operation ('training' or 'prediction').
        """
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'LoadValidate', mode)
        self.dbOperation = DatabaseOperation(self.run_id, self.data_path, mode)

    def values_from_schema(self, schema_file):
        """
        Method to read schema file and extract column names and the number of columns.

        Parameters
        ----------
        schema_file : str
            The name of the schema file (without extension) located in the 'apps/database' directory.

        Returns
        -------
        column_names : list
            A list of column names extracted from the schema file.
        number_of_columns : int
            The number of columns specified in the schema file.
        """
        try:
            self.logger.info('Start of Reading values From Schema...')
            with open('apps/database/'+schema_file+'.json', 'r') as f:
                dic = json.load(f)
                f.close()
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']
            self.logger.info('End of Reading values From Schema...')
        except ValueError:
            self.logger.exception('ValueError raised while Reading values From Schema')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while Reading values From Schema')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while Reading values From Schema: %s' % e)
            raise e
        return column_names, number_of_columns

    def validate_column_length(self, number_of_columns):
        """
    Method to validate the number of columns in the CSV files against the provided schema.

    Parameters
    ----------
    number_of_columns : int
        The expected number of columns in the CSV files.

    Raises
    ------
    OSError
        If an error occurs during file operations.
    Exception
        If an unexpected error occurs during the validation process.
    """
        try:
            self.logger.info('Start of Validating Column Length...')
            for file in listdir(self.data_path):
                self.logger.exception(f"{self.data_path + '/' + file}")
                csv = pd.read_csv(self.data_path+'/' + file)
                if csv.shape[1] == number_of_columns:
                    pass
                else:
                    shutil.move(self.data_path + '/' + file, self.data_path+'_rejects')
                    self.logger.info("Invalid Columns Length :: %s" % file)

            self.logger.info('End of Validating Column Length...')
        except OSError:
            self.logger.exception('OSError raised while Validating Column Length')
            raise OSError
        except Exception as e:
            self.logger.exception('Exception raised while Validating Column Length: %s' % e)
            raise e

    def validate_missing_values(self):
        """
        Method to validate if any column in the CSV files has all values missing.

        If all the values are missing in any column, the file is considered unsuitable for processing
        and is moved to a designated folder for rejected files.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If an error occurs during file operations.
        Exception
            If an unexpected error occurs during the validation process.
        """
        try:
            self.logger.info('Start of Validating Missing Values...')
            for file in listdir(self.data_path):
                if self._all_missing_values_in_any_column(file):
                    shutil.move(self.data_path + '/' + file, self.data_path + '_rejects')
                    self.logger.info("All Missing Values in Columns :: %s" % file)

            self.logger.info('End of Validating Missing Values...')
        except OSError:
            self.logger.exception('OSError raised while Validating Missing Values')
            raise OSError
        except Exception as e:
            self.logger.exception('Exception raised while Validating Missing Values: %s' % e)
            raise e

    def _all_missing_values_in_any_column(self, file):
        """
        Helper method to check if all values in any column of a CSV file are missing.

        Parameters
        ----------
        file : str
            The name of the CSV file to be checked.

        Returns
        -------
        bool
            True if all values are missing in any column, False otherwise.
        """
        try:
            csv = pd.read_csv(self.data_path + '/' + file)

            self.logger.exception(f"{self.data_path + '/' + file}")
            self.logger.exception(f"{csv.shape}")

            for column in csv:
                if (len(csv[column]) - csv[column].count()) == len(csv[column]):
                    return True
            return False
        except Exception as e:
            self.logger.exception('Exception occurred while checking missing values in file %s: %s' % (file, e))
            raise e

    def replace_missing_values(self):
        """
        Method to replace missing values in columns with "NULL".

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs during the replacement process.
        """
        try:
            self.logger.info('Start of Replacing Missing Values with NULL...')
            for file in os.listdir(self.data_path):
                self.logger.info('%s: We are in' % file)
                self.logger.info(f"What data is here {self.data_path}/{file}")

                if file.endswith(".csv"):
                    csv_path = os.path.join(self.data_path, file)
                    csv = pd.read_csv(csv_path)
                    csv.fillna('NULL', inplace=True)
                    csv.to_csv(csv_path, index=None, header=True)
                    self.logger.info('%s: File Transformed successfully!!' % file)
            self.logger.info('End of Replacing Missing Values with NULL...')
        except Exception as e:
            self.logger.exception('Exception raised while Replacing Missing Values with NULL: %s' % e)
            raise e

    def archive_old_files(self):
        """
        Method to archive old files from different directories.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs during the archiving process.
        """
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H%M%S")
        try:
            directories = ['_rejects', '_validation', '_processed', '_results']
            for directory in directories:
                source = os.path.join(self.data_path + directory)
                self.logger.info(f"{source}")
                if os.path.isdir(source):
                    archive_path = os.path.join(self.data_path + '_archive', directory[1:] + '_' + date_time_str)
                    os.makedirs(archive_path, exist_ok=True)
                    files = os.listdir(source)
                    for file in files:
                        destination = os.path.join(archive_path, file)
                        if file not in os.listdir(archive_path):
                            shutil.move(os.path.join(source, file), destination)
            self.logger.info('End of Archiving Old Files...')
        except Exception as e:
            self.logger.exception('Exception raised while Archiving Old Files: %s' % e)
            raise e

    def move_processed_files(self):
        """
        Move processed files to the processed directory.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs while moving processed files.
        """
        try:
            self.logger.info('Start of Moving Processed Files...')
            files = os.listdir(self.data_path)
            for file in files:
                src_path = os.path.join(self.data_path, file)
                dest_dir = self.data_path + '_processed'
                os.makedirs(dest_dir, exist_ok=True)  # Create destination directory if not exists
                dest_path = os.path.join(dest_dir, file)
                shutil.move(src_path, dest_path)
                self.logger.info("Moved the already processed file %s" % file)
            self.logger.info('End of Moving Processed Files...')
        except Exception as e:
            self.logger.exception('Exception raised while Moving Processed Files: %s' % e)
            raise e

    def validate_data_set(self, schema_name, table_name):
        """
        Validate a data set.

        This method performs several validation and transformation tasks on the provided data set.

        Parameters
        ----------
        schema_name : str
            The name of the schema to extract values from.
        table_name : str
            The name of the table to be created in the database.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs during data validation and transformation.
        """
        try:
            self.logger.info('Start of Data Load, validation, and transformation')
            # Archive old files
            self.archive_old_files()
            # Extract values from schema
            column_names, number_of_columns = self.values_from_schema(schema_name)
            # Validate column length in the file
            self.validate_column_length(number_of_columns)
            # Validate if any column has all values missing
            self.validate_missing_values()
            # Replace blanks in the CSV file with "Null" values
            self.replace_missing_values()
            # Create database with given name, if present open the connection! Create table with columns given in schema
            self.dbOperation.create_table(table_name, f'{table_name}_raw_data_t', column_names)
            # Insert CSV files in the table
            self.dbOperation.insert_data(table_name, f'{table_name}_raw_data_t')
            # Export data in table to CSV file
            self.dbOperation.export_csv(table_name, f'{table_name}_raw_data_t')
            # Move processed files
            self.move_processed_files()
            self.logger.info('Successful End of Data Load, validation, and transformation')
        except Exception:
            self.logger.exception('Unsuccessful End of Data Load, validation, and transformation')
            raise Exception

    def validate_train_set(self):
        """
        Validate the training data set.

        This method performs several validation and transformation tasks on the training data set.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs during data validation and transformation.
        """
        self.validate_data_set('schema_train', 'training')

    def validate_predict_set(self):
        """
        Validate the prediction data set.

        This method performs several validation and transformation tasks on the prediction data set.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an unexpected error occurs during data validation and transformation.
        """
        self.validate_data_set('schema_predict', 'prediction')

