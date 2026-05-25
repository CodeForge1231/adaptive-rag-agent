import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Configuration settings for the application.
    """
    def __init__(self):
        """
        Initialize settings by fetching values from the environment.
        """
        # Fetch the base API URL; fallback to local development URL
        self.base_api_url = os.getenv(
            "BASE_API_URL",
            "http://localhost:8000",
        )
        # Fetch the security token required for API authorization
        self.api_token = os.getenv(
            "API_TOKEN",
            "",
        )
