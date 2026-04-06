import logging
import os


class SecureLogger:
    """
    Simple logger for suspicious inputs and system events.
    """

    def __init__(self, log_file="logs/secure_ai.log"):
        os.makedirs("logs", exist_ok=True)

        self.logger = logging.getLogger("SecureAI")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)