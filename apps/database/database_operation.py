import os
import pandas as pd
from os import listdir

import sqlite3
import csv

import shutil

from apps.core.logger import Logger


class DatabaseOperation:
    """
    Class to handle database operations.

    Attributes
    ----------
    run_id : str
        The ID associated with the database operation run.
    data_path : str
        The path to the data.
    logger : Logger
        Logger instance for logging.
    file_name : str or None
        The name of the file.
    file_from_db : object or None
        The file obtained from the database.

    Methods
    -------
    __init__(run_id, data_path, mode)
        Initializes DatabaseOperation with run_id, data_path, and mode.
    """

    def __init__(self, run_id, data_path, mode):
        """
        Initializes DatabaseOperation with run_id, data_path, and mode.

        Parameters
        ----------
        run_id : str
            The ID associated with the database operation run.
        data_path : str
            The path to the data.
        mode : str
            The mode of operation.
        """
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'DatabaseOperation', mode)

        self.file_name = None
        self.file_from_db = None

    def database_connection(self, database_name):
        """
        Builds a connection to the specified database.

        Parameters
        ----------
        database_name : str
            The name of the database to connect to.

        Returns
        -------
        Connection
            Connection object representing the connection to the database.

        """
        try:
            conn = sqlite3.connect('apps/database/' + database_name + '.db')
            self.logger.info("Opened %s database successfully" % database_name)
        except ConnectionError as e:
            self.logger.info("Error while connecting to database: %s" % e)
            raise ConnectionError
        return conn

    def create_table(self, database_name, table_name, column_names):
        """Creates a table in the specified database, handling potential table existence.

        This method attempts to create a table named `table_name` with the provided
        `column_names` dictionary in the specified `database_name`. If the table already
        exists, it checks if any new columns need to be added based on the provided
        `column_names`.

    Args:
        database_name (str): The name of the database where the table will be created.
        table_name (str): The name of the table to be created.
        column_names (dict): A dictionary containing column names as keys and their
                             corresponding data types as values.

    Raises:
        Exception: If an error occurs during database operations.
    """
        try:
            self.logger.info('Start of Creating Table...')
            conn = self.database_connection(database_name)

            if database_name == 'prediction':
                conn.execute("DROP TABLE IF EXISTS '" + table_name + "';")

            cursor = conn.cursor()
            cursor.execute("SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name = '" + table_name + "'")
            if cursor.fetchone()[0] == 1:
                conn.close()
                self.logger.info('Tables created successfully')
                self.logger.info("Closed %s database successfully" % database_name)
            else:
                for key in column_names.keys():
                    _type = column_names[key]

                    # in try block we check if the table exists, if yes then add columns to the table
                    # else in catch block we will create the table --training_raw_data_t
                    try:
                        conn.execute(
                            "ALTER TABLE " + table_name + " ADD COLUMN {column_name} {dataType}".format(column_name=key,
                                                                                                        dataType=_type))
                        self.logger.info("ALTER TABLE " + table_name + " ADD COLUMN")
                    except Exception as e:
                        self.logger.info(f"{e}")
                        conn.execute(
                            "CREATE TABLE  " + table_name + " ({column_name} {dataType})".format(column_name=key,
                                                                                                 dataType=_type))
                        self.logger.info("CREATE TABLE " + table_name + " column_name")
                conn.close()
            self.logger.info('End of Creating Table...')
        except Exception as e:
            self.logger.exception(f'Exception raised while Creating Table: {e}')
            raise e

    def insert_data(self, database_name, table_name):
        """
        Inserts data into the specified table in the database.

        Parameters
        ----------
        database_name : str
            The name of the database where the data will be inserted.
        table_name : str
            The name of the table where the data will be inserted.

        Returns
        -------
        None

        """
        conn = self.database_connection(database_name)
        good_data_path = self.data_path

        self.logger.info(f"**Inserting data from file:** {good_data_path}")

        bad_data_path = self.data_path + '_rejects'
        only_files = [f for f in listdir(good_data_path)]
        self.logger.info('Start of Inserting Data into Table...')
        for file in only_files:
            try:
                # Log specific file path
                self.logger.info(f"**Inserting data from file:** {good_data_path + '/' + file}")
                inserted_data = pd.read_csv(f"{good_data_path + '/' + file}")
                data_shape = inserted_data.shape

                self.logger.info(f"Shape of Data to be inserted: {data_shape}")

                with open(good_data_path + '/' + file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter=",")
                    for line in enumerate(reader):
                        to_db = ''
                        for list_ in (line[1]):
                            try:
                                to_db = to_db + ",'" + list_ + "'"
                            except Exception as e:
                                raise e
                        to_db = to_db.lstrip(',')
                        conn.execute("INSERT INTO " + table_name + " values ({values})".format(values=to_db))
                        conn.commit()

            except Exception as e:
                conn.rollback()
                self.logger.exception('Exception raised while Inserting Data into Table: %s ' % e)
                shutil.move(good_data_path + '/' + file, bad_data_path)
                conn.close()
        conn.close()
        self.logger.info("What is the data path?")
        self.logger.info('End of Inserting Data into Table...')

    def export_csv(self, database_name, table_name):
        """
        Exports data from the specified table in the database to a CSV file.

        Parameters
        ----------
        database_name : str
            The name of the database from which data will be exported.
        table_name : str
            The name of the table from which data will be exported.

        Returns
        -------
        None

        """
        self.file_from_db = self.data_path + str('_validation/')
        self.file_name = 'InputFile.csv'
        try:
            self.logger.info('Start of Exporting Data into CSV...')
            conn = self.database_connection(database_name)
            sql_select = "SELECT * FROM " + table_name
            cursor = conn.cursor()
            cursor.execute(sql_select)
            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]
            # Make the CSV output directory
            if not os.path.isdir(self.file_from_db):
                os.makedirs(self.file_from_db)
            # Open CSV file for writing.
            csv_file = csv.writer(open(self.file_from_db + self.file_name, 'w', newline=''), delimiter=',',
                                  lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')
            # Add the headers and data to the CSV file.
            csv_file.writerow(headers)
            csv_file.writerows(results)

            inserted_data = pd.read_csv(self.file_from_db + self.file_name)
            data_shape = inserted_data.shape

            self.logger.info(f"Shape of Exported Data: {data_shape}")
            self.logger.info('End of Exporting Data into CSV...')
        except Exception as e:
            self.logger.exception('Exception raised while Exporting Data into CSV: %s ' % e)

