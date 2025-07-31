"""Enhanced Analytics Service with Google Sheets integration."""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import pandas as pd

from ..api.ga4_api import GA4Client
from ..api.sheets_api import SheetsClient

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Enhanced Analytics Service with GA4 and Google Sheets integration."""
    
    def __init__(self, ga_client: GA4Client, sheets_client: SheetsClient):
        """Initialize Analytics Service.
        
        Args:
            ga_client: GA4 API client
            sheets_client: Google Sheets API client
        """
        self.ga_client = ga_client
        self.sheets_client = sheets_client
        
    def run_monthly_report(
        self,
        property_id: str,
        sheet_config: Dict[str, str],
        metrics: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run monthly GA4 report and export to Google Sheets.
        
        Args:
            property_id: GA4 property ID
            sheet_config: Sheet configuration (sheet_id, worksheet_name)
            metrics: List of GA4 metrics (uses defaults if None)
            dimensions: List of GA4 dimensions (uses defaults if None)
            
        Returns:
            Dict with report results
        """
        try:
            # Get last month date range
            today = date.today()
            first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_last_month = today.replace(day=1) - timedelta(days=1)
            
            start_date = first_day_last_month.strftime("%Y-%m-%d")
            end_date = last_day_last_month.strftime("%Y-%m-%d")
            
            # Default metrics for monthly report
            if not metrics:
                metrics = [
                    "activeUsers",
                    "newUsers",
                    "sessions", 
                    "screenPageViews",
                    "userEngagementDuration",
                    "bounceRate"
                ]
            
            # Default dimensions (minimal for monthly summary)
            if not dimensions:
                dimensions = []
                
            logger.info(f"Running monthly report for {start_date} to {end_date}")
            
            # Get GA4 data
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
                return {
                    "status": "no_data",
                    "date_range": {"start": start_date, "end": end_date, "label": first_day_last_month.strftime('%B %Y')}
                }
            
            # Add report metadata
            df["report_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df["date_range"] = f"{start_date} to {end_date}"
            df["report_type"] = "monthly"
            
            # Export to Google Sheets (append mode for monthly reports)
            sheets_result = self.sheets_client.dataframe_to_sheets(
                df=df,
                spreadsheet_id=sheet_config["sheet_id"],
                worksheet_name=sheet_config.get("worksheet_name", "raw_data"),
                insert_mode="append",
                include_header=True
            )
            
            logger.info(f"Monthly report completed successfully: {sheets_result}")
            
            return {
                "status": "success",
                "date_range": {
                    "start": start_date,
                    "end": end_date, 
                    "label": first_day_last_month.strftime('%B %Y')
                },
                "rows_processed": len(df),
                "sheets_result": sheets_result
            }
            
        except Exception as e:
            logger.error(f"Error in monthly report: {e}")
            raise
    
    def run_daily_report(
        self,
        property_id: str,
        sheet_config: Dict[str, str], 
        days: int = 7,
        metrics: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run daily GA4 report and export to Google Sheets.
        
        Args:
            property_id: GA4 property ID
            sheet_config: Sheet configuration (sheet_id, worksheet_name)
            days: Number of days to include in report
            metrics: List of GA4 metrics (uses defaults if None)
            dimensions: List of GA4 dimensions (uses defaults if None)
            
        Returns:
            Dict with report results
        """
        try:
            # Get date range for last N days
            today = date.today()
            start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
            
            # Default metrics for daily report
            if not metrics:
                metrics = [
                    "activeUsers",
                    "sessions",
                    "screenPageViews",
                    "userEngagementDuration",
                    "bounceRate"
                ]
            
            # Default dimensions (include date for daily breakdown)
            if not dimensions:
                dimensions = ["date"]
                
            logger.info(f"Running daily report for {start_date} to {end_date}")
            
            # Get GA4 data
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
                return {
                    "status": "no_data",
                    "date_range": {"start": start_date, "end": end_date}
                }
            
            # Sort by date (most recent first for top insertion)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                df = df.sort_values("date", ascending=False)
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            
            # Add report metadata
            df["report_generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df["report_type"] = "daily"
            
            # Export to Google Sheets (insert at top for daily reports)
            sheets_result = self.sheets_client.dataframe_to_sheets(
                df=df,
                spreadsheet_id=sheet_config["sheet_id"],
                worksheet_name=sheet_config.get("worksheet_name", "daily_metrics"),
                insert_mode="top",
                include_header=True
            )
            
            logger.info(f"Daily report completed successfully: {sheets_result}")
            
            return {
                "status": "success",
                "date_range": {
                    "start": start_date,
                    "end": end_date,
                    "days": days
                },
                "rows_count": len(df),
                "sheets_result": sheets_result
            }
            
        except Exception as e:
            logger.error(f"Error in daily report: {e}")
            raise
    
    def get_property_metadata(self, property_id: str) -> Dict[str, Any]:
        """Get available metrics and dimensions for GA4 property.
        
        Args:
            property_id: GA4 property ID
            
        Returns:
            Dict with available metrics and dimensions
        """
        try:
            return self.ga_client.get_metadata(property_id)
        except Exception as e:
            logger.error(f"Error getting property metadata: {e}")
            raise
    
    def export_dataframe_to_sheets(
        self,
        df: pd.DataFrame,
        sheet_config: Dict[str, str],
        insert_mode: str = "append"
    ) -> Dict[str, Any]:
        """Export any DataFrame to Google Sheets.
        
        Args:
            df: DataFrame to export
            sheet_config: Sheet configuration
            insert_mode: 'append' or 'top'
            
        Returns:
            Dict with export results  
        """
        try:
            return self.sheets_client.dataframe_to_sheets(
                df=df,
                spreadsheet_id=sheet_config["sheet_id"],
                worksheet_name=sheet_config.get("worksheet_name", "data"),
                insert_mode=insert_mode,
                include_header=True
            )
        except Exception as e:
            logger.error(f"Error exporting DataFrame to Sheets: {e}")
            raise