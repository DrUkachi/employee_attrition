import logging


class Logger:
    """
    Class to generate logs.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for logging.

    Methods
    -------
    __init__(run_id, log_module, log_file_name)
        Initializes Logger with run_id, log_module, and log_file_name.
    info(message)
        Logs an informational message.
    exception(message)
        Logs an exception message.

    Version History
    ---------------
    uosisiogu       15-MAY-2020    1.0      Initial creation
    """

    def __init__(self, run_id, log_module, log_file_name):
        """
        Initializes Logger with run_id, log_module, and log_file_name.

        Parameters
        ----------
        run_id : str
            The ID associated with the logging run.
        log_module : str
            The name of the module being logged.
        log_file_name : str
            The name of the log file.
        """
        self.logger = logging.getLogger(str(log_module)+'_' + str(run_id))
        self.logger.setLevel(logging.DEBUG)
        if log_file_name == 'training':
            file_handler = logging.FileHandler('logs/train_logs/train_log_' + str(run_id) + '.log')
        else:
            file_handler = logging.FileHandler('logs/predict_logs/predict_log_' + str(run_id) + '.log')

        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        """
        Logs an informational message.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        self.logger.info(message)

    def exception(self, message):
        """
        Logs an exception message.

        Parameters
        ----------
        message : str
            The exception message to be logged.
        """
        self.logger.exception(message)
