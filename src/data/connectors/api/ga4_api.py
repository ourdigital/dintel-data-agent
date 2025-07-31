"""Google Analytics 4 API client wrapper."""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)

logger = logging.getLogger(__name__)

class GA4Client:
    """Google Analytics 4 API client wrapper."""
    
    def __init__(self, client: BetaAnalyticsDataClient):
        """Initialize GA4 client.
        
        Args:
            client: Authenticated BetaAnalyticsDataClient instance
        """
        self.client = client
        
    def run_report(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        metrics: List[str],
        dimensions: Optional[List[str]] = None,
        limit: int = 100000
    ) -> Dict[str, Any]:
        """Run a GA4 report.
        
        Args:
            property_id: GA4 property ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            metrics: List of metric names
            dimensions: List of dimension names
            limit: Maximum number of rows to return
            
        Returns:
            Dict containing report data and metadata
        """
        try:
            # Build request
            date_ranges = [DateRange(start_date=start_date, end_date=end_date)]
            metric_objects = [Metric(name=metric) for metric in metrics]
            dimension_objects = [Dimension(name=dim) for dim in (dimensions or [])]
            
            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=date_ranges,
                metrics=metric_objects,
                dimensions=dimension_objects,
                limit=limit
            )
            
            # Execute request
            response = self.client.run_report(request)
            
            logger.info(f"GA4 report executed successfully. Rows: {len(response.rows)}")
            
            # Process response
            return self._process_response(response, metrics, dimensions or [])
            
        except Exception as e:
            logger.error(f"Error running GA4 report: {e}")
            raise
    
    def _process_response(
        self, 
        response, 
        metrics: List[str], 
        dimensions: List[str]
    ) -> Dict[str, Any]:
        """Process GA4 API response.
        
        Args:
            response: GA4 API response
            metrics: List of metric names
            dimensions: List of dimension names
            
        Returns:
            Dict with processed data and metadata
        """
        rows_data = []
        
        for row in response.rows:
            row_data = {}
            
            # Extract dimension values
            for i, dimension_value in enumerate(row.dimension_values):
                if i < len(dimensions):
                    row_data[dimensions[i]] = dimension_value.value
            
            # Extract metric values
            for i, metric_value in enumerate(row.metric_values):
                if i < len(metrics):
                    # Try to convert to numeric
                    try:
                        row_data[metrics[i]] = float(metric_value.value)
                    except (ValueError, TypeError):
                        row_data[metrics[i]] = metric_value.value
            
            rows_data.append(row_data)
        
        return {
            "data": rows_data,
            "row_count": len(rows_data),
            "property_id": response.property_quota.property_id if response.property_quota else None,
            "metadata": {
                "metrics": metrics,
                "dimensions": dimensions,
                "total_rows": len(response.rows)
            }
        }
    
    def get_metadata(self, property_id: str) -> Dict[str, List[str]]:
        """Get available metrics and dimensions for a property.
        
        Args:
            property_id: GA4 property ID
            
        Returns:
            Dict with available metrics and dimensions
        """
        try:
            # This would require additional API calls in a real implementation
            # For now, return common metrics and dimensions
            return {
                "metrics": [
                    "activeUsers",
                    "newUsers", 
                    "sessions",
                    "screenPageViews",
                    "bounceRate",
                    "averageSessionDuration",
                    "userEngagementDuration",
                    "eventCount"
                ],
                "dimensions": [
                    "date",
                    "country",
                    "city",
                    "deviceCategory",
                    "operatingSystem",
                    "browser",
                    "sourceMedium",
                    "sessionDefaultChannelGroup"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting GA4 metadata: {e}")
            raise