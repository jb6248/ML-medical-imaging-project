import os
import logging


class ExpLogger:
    def __init__(self, log_file_path):
        os.makedirs(
            os.path.dirname(log_file_path), exist_ok=True
        )  # Ensure the log directory exists

        # Create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # File handler to log messages to a file
        self.file_handler = logging.FileHandler(log_file_path, mode="w")
        self.file_handler.setLevel(logging.INFO)

        # Console handler to logger.info messages to the console
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        # Formatter for consistent logging output
        self.formatter = logging.Formatter('')
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def info(self, msg):
        self.logger.info(msg)
