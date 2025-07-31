"""Enhanced Google Analytics connector with Google Sheets integration."""
import os
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime, timedelta
import yaml

from .auth.google_auth import GoogleAuth
from .api.ga4_api import GA4Client
from .api.sheets_api import SheetsClient
from .services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

class GoogleAnalyticsEnhancedConnector:
    """Enhanced Google Analytics connector with advanced features."""
    
    def __init__(self, 
                 credentials_path: str = "config/api_credentials.yaml",
                 config_path: str = "config/pipeline_config.yaml"):
        """Initialize Enhanced Google Analytics Connector.
        
        Args:
            credentials_path: API credentials file path
            config_path: Pipeline configuration file path
        """
        self.credentials_path = credentials_path
        self.config_path = config_path
        self.config = self._load_config()
        self.credentials = self._load_credentials()
        
        # Initialize enhanced components
        self.auth = GoogleAuth()
        self.ga_client = GA4Client(self.auth.get_ga_client())
        self.sheets_client = SheetsClient(self.auth.get_sheets_client())
        self.analytics_service = AnalyticsService(self.ga_client, self.sheets_client)
        
        logger.info("Enhanced GA connector initialized successfully")
        
    def _load_credentials(self) -> Dict[str, Any]:
        """Load API credentials."""
        try:
            with open(self.credentials_path, 'r', encoding='utf-8') as f:
                credentials = yaml.safe_load(f)
            return credentials
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_date_range(self, range_type: Optional[str] = None) -> tuple:
        """Get date range based on configuration.
        
        Args:
            range_type: Override default date range type
            
        Returns:
            Tuple of (start_date, end_date) as YYYY-MM-DD strings
        """
        date_range = range_type or self.config['collection']['default_date_range']
        today = datetime.now().date()
        
        if date_range == "last_7_days":
            start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "last_30_days":
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "last_90_days":
            start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif date_range == "custom":
            custom_range = self.config['collection']['custom_date_range']
            start_date = custom_range['start_date']
            end_date = custom_range['end_date']
        else:
            # Default: last 30 days
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
        logger.info(f"Date range set: {start_date} ~ {end_date}")
        return start_date, end_date
    
    def fetch_data(self, 
                   custom_metrics: Optional[List[str]] = None,
                   custom_dimensions: Optional[List[str]] = None,
                   date_range: Optional[str] = None) -> pd.DataFrame:
        """Fetch Google Analytics data with enhanced features.
        
        Args:
            custom_metrics: Override default metrics
            custom_dimensions: Override default dimensions  
            date_range: Override default date range
            
        Returns:
            pd.DataFrame with GA data
        """
        try:
            ga_config = self.config['collection']['sources']['google_analytics']
            
            if not ga_config['enabled']:
                logger.info("Google Analytics data collection is disabled")
                return pd.DataFrame()
            
            # Get configuration
            property_id = self.credentials['google_analytics']['property_id']
            metrics = custom_metrics or ga_config['metrics']
            dimensions = custom_dimensions or ga_config['dimensions']
            start_date, end_date = self._get_date_range(date_range)
            
            # Fetch data using enhanced GA4 client
            report_data = self.ga_client.run_report(
                property_id=property_id,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
                dimensions=dimensions
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(report_data["data"])
            
            if df.empty:
                logger.warning("No data returned from GA4")
                return df
            
            # Enhanced data processing
            df = self._process_dataframe(df, metrics, dimensions)
            
            logger.info(f"Successfully fetched {len(df)} rows of GA4 data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching GA4 data: {e}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame, metrics: List[str], dimensions: List[str]) -> pd.DataFrame:
        """Enhanced DataFrame processing."""
        try:
            # Convert date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                df = df.sort_values('date', ascending=False)
                
            # Convert numeric metrics
            for metric in metrics:
                if metric in df.columns:
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # Add metadata
            df['fetched_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame: {e}")
            return df
    
    def run_monthly_report(self, 
                          export_to_sheets: bool = True,
                          custom_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run monthly report with Google Sheets export.
        
        Args:
            export_to_sheets: Whether to export to Google Sheets
            custom_metrics: Override default metrics
            
        Returns:
            Dict with report results
        """
        try:
            if not export_to_sheets:
                # Just fetch data without Sheets export
                df = self.fetch_data(custom_metrics=custom_metrics, date_range="last_30_days")
                return {
                    "status": "success",
                    "data": df,
                    "rows": len(df)
                }
            
            # Use enhanced analytics service for Sheets integration
            property_id = self.credentials['google_analytics']['property_id']
            sheet_config = {
                "sheet_id": os.getenv("SHEET_ID"),
                "worksheet_name": os.getenv("SHEET_NAME", "raw_data")
            }
            
            return self.analytics_service.run_monthly_report(
                property_id=property_id,
                sheet_config=sheet_config,
                metrics=custom_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in monthly report: {e}")
            raise
    
    def run_daily_report(self, 
                        days: int = 7,
                        export_to_sheets: bool = True,
                        custom_metrics: Optional[List[str]] = None,
                        custom_dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run daily report with Google Sheets export.
        
        Args:
            days: Number of days to include
            export_to_sheets: Whether to export to Google Sheets
            custom_metrics: Override default metrics
            custom_dimensions: Override default dimensions
            
        Returns:
            Dict with report results
        """
        try:
            if not export_to_sheets:
                # Just fetch data without Sheets export
                df = self.fetch_data(
                    custom_metrics=custom_metrics,
                    custom_dimensions=custom_dimensions or ["date"],
                    date_range="custom"
                )
                return {
                    "status": "success", 
                    "data": df,
                    "rows": len(df)
                }
            
            # Use enhanced analytics service for Sheets integration
            property_id = self.credentials['google_analytics']['property_id']
            sheet_config = {
                "sheet_id": os.getenv("SHEET_ID"),
                "worksheet_name": "daily_metrics"
            }
            
            return self.analytics_service.run_daily_report(
                property_id=property_id,
                sheet_config=sheet_config,
                days=days,
                metrics=custom_metrics,
                dimensions=custom_dimensions or ["date"]
            )
            
        except Exception as e:
            logger.error(f"Error in daily report: {e}")
            raise
    
    def export_to_sheets(self, 
                        df: pd.DataFrame,
                        worksheet_name: str = "data",
                        insert_mode: str = "append") -> Dict[str, Any]:
        """Export DataFrame to Google Sheets.
        
        Args:
            df: DataFrame to export
            worksheet_name: Target worksheet name
            insert_mode: 'append' or 'top'
            
        Returns:
            Dict with export results
        """
        try:
            sheet_config = {
                "sheet_id": os.getenv("SHEET_ID"),
                "worksheet_name": worksheet_name
            }
            
            return self.analytics_service.export_dataframe_to_sheets(
                df=df,
                sheet_config=sheet_config,
                insert_mode=insert_mode
            )
            
        except Exception as e:
            logger.error(f"Error exporting to Sheets: {e}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str = "data/raw/google_analytics/") -> str:
        """Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Output directory path
            
        Returns:
            str: Saved file path
        """
        if df.empty:
            logger.warning("No data to save")
            return ""
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Add current date to filename
        current_date = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(output_path, f"ga_data_{current_date}.csv")
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"GA data saved to CSV: {file_path}")
        
        return file_path


# Maintain backwards compatibility
class GoogleAnalyticsConnector(GoogleAnalyticsEnhancedConnector):
    """Legacy connector class for backwards compatibility."""
    pass