"""Google services authentication module."""
import os
from typing import Any, Dict

from dotenv import load_dotenv
from google.oauth2 import service_account
import gspread
from google.analytics.data_v1beta import BetaAnalyticsDataClient

load_dotenv()

class GoogleAuth:
    """Handles authentication for various Google services."""
    
    def __init__(self):
        """Initialize with environment variables."""
        self.scopes = os.getenv("SCOPES", "").split(",")
        self.service_account_file = os.getenv("SERVICE_ACCOUNT_FILE")
        self.credentials = self._get_credentials()
    
    def _get_credentials(self):
        """Get service account credentials."""
        try:
            if not self.service_account_file:
                raise ValueError("SERVICE_ACCOUNT_FILE environment variable not set")
            
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=self.scopes
            )
            return credentials
        except Exception as e:
            print(f"Error loading credentials: {e}")
            raise
    
    def get_ga_client(self):
        """Initialize and return Google Analytics Data client."""
        try:
            return BetaAnalyticsDataClient(credentials=self.credentials)
        except Exception as e:
            print(f"Error initializing GA client: {e}")
            raise
    
    def get_sheets_client(self):
        """Initialize and return Google Sheets client."""
        try:
            return gspread.authorize(self.credentials)
        except Exception as e:
            print(f"Error initializing Sheets client: {e}")
            raise