from datetime import datetime
import secrets


class Config:
    """
    Class for configuration instance attributes.

    Attributes
    ----------
    training_data_path : str
        Path to the training data.
    training_database : str
        Database for training.
    prediction_data_path : str
        Path to the prediction data.
    prediction_database : str
        Database for prediction.
    """

    def __init__(self):
        """
        Initializes the Config class with default values for attributes.
        """
        self.current_time = None
        self.date = None
        self.now = None
        self.training_data_path = 'data/train_data'
        self.training_database = 'training'
        self.prediction_data_path = 'data/predict_data'
        self.prediction_database = 'prediction'

    def get_run_id(self):
        """
        Generates a unique run ID.

        Returns
        -------
        str
            A unique run ID.
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H%M%S")
        random_part = secrets.token_hex(8)  # Generate 8-byte hex string using secrets.token_hex
        return str(self.date) + "_" + str(self.current_time) + "_" + random_part
