"""Google Sheets API client wrapper."""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import gspread
from gspread.exceptions import WorksheetNotFound

logger = logging.getLogger(__name__)

class SheetsClient:
    """Google Sheets API client wrapper."""
    
    def __init__(self, client: gspread.Client):
        """Initialize Sheets client.
        
        Args:
            client: Authenticated gspread Client instance
        """
        self.client = client
        
    def get_spreadsheet(self, spreadsheet_id: str):
        """Get spreadsheet by ID.
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            
        Returns:
            gspread Spreadsheet object
        """
        try:
            return self.client.open_by_key(spreadsheet_id)
        except Exception as e:
            logger.error(f"Error opening spreadsheet {spreadsheet_id}: {e}")
            raise
    
    def get_or_create_worksheet(
        self, 
        spreadsheet_id: str, 
        worksheet_name: str,
        rows: int = 1000,
        cols: int = 26
    ):
        """Get existing worksheet or create new one.
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet
            rows: Number of rows for new worksheet
            cols: Number of columns for new worksheet
            
        Returns:
            gspread Worksheet object
        """
        try:
            spreadsheet = self.get_spreadsheet(spreadsheet_id)
            
            try:
                # Try to get existing worksheet
                worksheet = spreadsheet.worksheet(worksheet_name)
                logger.info(f"Using existing worksheet: {worksheet_name}")
                return worksheet
                
            except WorksheetNotFound:
                # Create new worksheet if it doesn't exist
                worksheet = spreadsheet.add_worksheet(
                    title=worksheet_name,
                    rows=rows,
                    cols=cols
                )
                logger.info(f"Created new worksheet: {worksheet_name}")
                return worksheet
                
        except Exception as e:
            logger.error(f"Error accessing worksheet {worksheet_name}: {e}")
            raise
    
    def append_data(
        self, 
        spreadsheet_id: str, 
        worksheet_name: str, 
        data: List[List[Any]],
        value_input_option: str = "USER_ENTERED"
    ) -> Dict[str, Any]:
        """Append data to worksheet.
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet
            data: Data to append as list of lists
            value_input_option: How to interpret input values
            
        Returns:
            Dict with operation result
        """
        try:
            worksheet = self.get_or_create_worksheet(spreadsheet_id, worksheet_name)
            
            # Append data
            result = worksheet.append_rows(data, value_input_option=value_input_option)
            
            logger.info(f"Row appended to {worksheet_name} in spreadsheet {spreadsheet_id}")
            
            return {
                "status": "success",
                "rows_added": len(data),
                "spreadsheet_id": spreadsheet_id,
                "range": result.get("tableRange", "")
            }
            
        except Exception as e:
            logger.error(f"Error appending data to {worksheet_name}: {e}")
            raise
    
    def insert_data_at_top(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        data: List[List[Any]],
        has_header: bool = True
    ) -> Dict[str, Any]:
        """Insert data at the top of worksheet (after header if present).
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet  
            data: Data to insert as list of lists
            has_header: Whether worksheet has header row
            
        Returns:
            Dict with operation result
        """
        try:
            worksheet = self.get_or_create_worksheet(spreadsheet_id, worksheet_name)
            
            # Get current data
            current_data = worksheet.get_all_values()
            
            if not current_data and data:
                # Empty sheet, just append data
                return self.append_data(spreadsheet_id, worksheet_name, data)
            
            # Clear worksheet
            worksheet.clear()
            
            # Rebuild data with new rows at top
            new_data = []
            
            if has_header and current_data:
                # Keep header row
                header = current_data[0]
                new_data.append(header)
                
                # Add new data
                new_data.extend(data)
                
                # Add existing data (skip header)
                if len(current_data) > 1:
                    new_data.extend(current_data[1:])
            else:
                # No header, just add new data then existing
                new_data.extend(data)
                new_data.extend(current_data)
            
            # Update worksheet with new data
            if new_data:
                worksheet.update(new_data)
            
            logger.info(f"{len(data)} rows inserted at top of {worksheet_name} in spreadsheet {spreadsheet_id}")
            
            return {
                "status": "success", 
                "rows_added": len(data),
                "total_rows": len(new_data),
                "spreadsheet_id": spreadsheet_id
            }
            
        except Exception as e:
            logger.error(f"Error inserting data at top of {worksheet_name}: {e}")
            raise
    
    def dataframe_to_sheets(
        self,
        df: pd.DataFrame,
        spreadsheet_id: str,
        worksheet_name: str,
        insert_mode: str = "append",
        include_header: bool = True
    ) -> Dict[str, Any]:
        """Export DataFrame to Google Sheets.
        
        Args:
            df: Pandas DataFrame to export
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet
            insert_mode: 'append' or 'top'
            include_header: Whether to include DataFrame column names
            
        Returns:
            Dict with operation result
        """
        try:
            if df.empty:
                logger.warning("DataFrame is empty, nothing to export")
                return {"status": "skipped", "reason": "empty_dataframe"}
            
            # Convert DataFrame to list of lists
            data = []
            
            if include_header:
                data.append(df.columns.tolist())
            
            # Add data rows
            for _, row in df.iterrows():
                data.append(row.tolist())
            
            # Export based on insert mode
            if insert_mode == "top":
                return self.insert_data_at_top(
                    spreadsheet_id, 
                    worksheet_name, 
                    data[1:] if include_header else data,  # Skip header for insert_at_top
                    has_header=True
                )
            else:
                return self.append_data(spreadsheet_id, worksheet_name, data)
                
        except Exception as e:
            logger.error(f"Error exporting DataFrame to Sheets: {e}")
            raise