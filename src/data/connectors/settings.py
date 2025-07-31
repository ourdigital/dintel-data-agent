"""Enhanced settings and configuration for data connectors."""
import os
from typing import Dict, List, Any
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API configuration
GA_PROPERTY_ID = os.getenv("PROPERTY_ID")
SHEETS_ID = os.getenv("SHEET_ID")
SHEETS_NAME = os.getenv("SHEET_NAME", "raw_data")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SCOPES = os.getenv("SCOPES", "").split(",")

# Enhanced default metrics for Google Analytics
DEFAULT_GA_METRICS = [
    "activeUsers",
    "newUsers",
    "eventCount",
    "userEngagementDuration",
    "screenPageViews",
    "sessions",
    "averageSessionDuration"
]

# Enhanced default dimensions for Google Analytics
DEFAULT_GA_DIMENSIONS = [
    "date",
    "deviceCategory",
    "country"
]

def get_last_month_date_range() -> Dict[str, str]:
    """Get the date range for the previous month.
    
    Returns:
        Dict with start_date and end_date as YYYY-MM-DD strings
    """
    today = date.today()
    first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day_last_month = today.replace(day=1) - timedelta(days=1)
    
    return {
        "start_date": first_day_last_month.strftime("%Y-%m-%d"),
        "end_date": last_day_last_month.strftime("%Y-%m-%d"),
        "label": f"{first_day_last_month.strftime('%B %Y')}"
    }

def get_date_range(days: int = 30) -> Dict[str, str]:
    """Get a date range for the last N days.
    
    Args:
        days: Number of days to go back
        
    Returns:
        Dict with start_date and end_date as YYYY-MM-DD strings
    """
    today = date.today()
    start_date = today - timedelta(days=days)
    
    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "label": f"Last {days} days"
    }

def get_historical_month_range(year: int = 2024, month: int = 11) -> Dict[str, str]:
    """Get a specific historical month date range.
    
    Args:
        year: Year for the report
        month: Month for the report (1-12)
        
    Returns:
        Dict with start_date and end_date as YYYY-MM-DD strings
    """
    from calendar import monthrange
    
    first_day = date(year, month, 1)
    last_day = date(year, month, monthrange(year, month)[1])
    
    return {
        "start_date": first_day.strftime("%Y-%m-%d"),
        "end_date": last_day.strftime("%Y-%m-%d"),
        "label": f"{first_day.strftime('%B %Y')}"
    }