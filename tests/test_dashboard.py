import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import yaml
from pathlib import Path
import sys
from datetime import date

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app.dashboard import Dashboard # Module to be tested

class TestDashboard(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file content
        self.dummy_config_content = {
            'app': {
                'title': 'Test Dashboard',
                'theme': {'primary_color': '#000000'}
            },
            'database': {
                'type': 'sqlite',
                'path': 'dummy.db'
            }
        }
        # Path for a dummy config file
        self.dummy_config_path = project_root / "tests" / "dummy_config.yaml"
        
        # Write the dummy config to a temporary file for _load_config tests
        with open(self.dummy_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dummy_config_content, f)

    def tearDown(self):
        # Clean up the dummy config file
        if self.dummy_config_path.exists():
            self.dummy_config_path.unlink()

    @patch('src.app.dashboard.DatabaseManager') # Mock DatabaseManager
    @patch('src.app.dashboard.st') # Mock streamlit
    def test_dashboard_init_default_config_path(self, mock_st, MockDatabaseManager):
        print("Running test_dashboard_init_default_config_path")
        # For this test, we assume the default config path might not exist or be readable
        # So, we also mock _load_config to prevent it from trying to load a real file
        with patch.object(Dashboard, '_load_config', return_value=self.dummy_config_content) as mock_load_config:
            dashboard = Dashboard()
            expected_path = project_root / "config" / "pipeline_config.yaml"
            self.assertEqual(dashboard.config_path, expected_path)
            self.assertFalse(dashboard.using_sample_data)
            MockDatabaseManager.assert_called_once_with(str(expected_path))
            mock_load_config.assert_called_once() # Ensure _load_config was called
            mock_st.set_page_config.assert_called() # Check if setup_page was implicitly called
        print("Finished test_dashboard_init_default_config_path")

    @patch('src.app.dashboard.DatabaseManager')
    @patch('src.app.dashboard.st')
    def test_dashboard_init_provided_config_path_str(self, mock_st, MockDatabaseManager):
        print("Running test_dashboard_init_provided_config_path_str")
        with patch.object(Dashboard, '_load_config', return_value=self.dummy_config_content):
            test_path_str = "custom/config.yaml"
            dashboard = Dashboard(config_path=test_path_str)
            self.assertEqual(dashboard.config_path, Path(test_path_str))
            self.assertFalse(dashboard.using_sample_data)
            MockDatabaseManager.assert_called_once_with(test_path_str)
        print("Finished test_dashboard_init_provided_config_path_str")

    @patch('src.app.dashboard.DatabaseManager')
    @patch('src.app.dashboard.st')
    def test_dashboard_init_provided_config_path_pathlib(self, mock_st, MockDatabaseManager):
        print("Running test_dashboard_init_provided_config_path_pathlib")
        with patch.object(Dashboard, '_load_config', return_value=self.dummy_config_content):
            test_path_obj = Path("custom/pathlib/config.yaml")
            dashboard = Dashboard(config_path=test_path_obj)
            self.assertEqual(dashboard.config_path, test_path_obj)
            self.assertFalse(dashboard.using_sample_data)
            MockDatabaseManager.assert_called_once_with(str(test_path_obj))
        print("Finished test_dashboard_init_provided_config_path_pathlib")
    
    @patch('src.app.dashboard.st') # Mock streamlit for setup_page
    def test_load_config_success(self, mock_st):
        print("Running test_load_config_success")
        # We use the dummy_config_path created in setUp
        # No need to mock DatabaseManager here as _load_config doesn't use it
        dashboard = Dashboard(config_path=self.dummy_config_path) # This will call _load_config
        self.assertEqual(dashboard.config, self.dummy_config_content)
        print("Finished test_load_config_success")

    @patch('src.app.dashboard.st') # Mock streamlit for setup_page and error display
    @patch('src.app.dashboard.logger') # Mock logger to check error logging
    def test_load_config_file_not_found(self, mock_logger, mock_st):
        print("Running test_load_config_file_not_found")
        # Path to a non-existent config file
        non_existent_path = project_root / "tests" / "non_existent_config.yaml"
        
        # Temporarily mock DatabaseManager for __init__ as it's not the focus here
        with patch('src.app.dashboard.DatabaseManager'):
            dashboard = Dashboard(config_path=non_existent_path) # This will call _load_config

        self.assertEqual(dashboard.config, {}) # Should return empty dict on failure
        mock_logger.error.assert_called_once()
        mock_st.error.assert_called_once()
        print("Finished test_load_config_file_not_found")

    @patch('src.app.dashboard.st') # Mock streamlit for setup_page
    @patch.object(Dashboard, '_load_config', return_value={}) # Mock _load_config directly
    def test_get_data_success(self, mock_load_config, mock_st):
        print("Running test_get_data_success")
        dashboard = Dashboard(config_path=self.dummy_config_path)
        dashboard.db_manager = MagicMock() # Mock the db_manager instance
        
        mock_df = pd.DataFrame({'data': ['live']})
        dashboard.db_manager.read_sql_query.return_value = mock_df
        
        start = date(2023, 1, 1)
        end = date(2023, 1, 31)
        
        df_result = dashboard.get_data(start, end, ['source1'])
        
        self.assertTrue(df_result.equals(mock_df))
        self.assertFalse(dashboard.using_sample_data)
        dashboard.db_manager.read_sql_query.assert_called_once()
        print("Finished test_get_data_success")

    @patch('src.app.dashboard.st') # Mock streamlit for setup_page and error display
    @patch.object(Dashboard, '_load_config', return_value={}) # Mock _load_config directly
    @patch.object(Dashboard, 'load_sample_data') # Mock load_sample_data
    def test_get_data_db_exception(self, mock_load_sample_data, mock_load_config, mock_st):
        print("Running test_get_data_db_exception")
        dashboard = Dashboard(config_path=self.dummy_config_path)
        dashboard.db_manager = MagicMock()
        dashboard.db_manager.read_sql_query.side_effect = Exception("DB Error")
        
        mock_sample_df = pd.DataFrame({'data': ['sample']})
        mock_load_sample_data.return_value = mock_sample_df
        
        start = date(2023, 1, 1)
        end = date(2023, 1, 31)
        
        df_result = dashboard.get_data(start, end, ['source1'])
        
        self.assertTrue(df_result.equals(mock_sample_df))
        self.assertTrue(dashboard.using_sample_data)
        dashboard.db_manager.read_sql_query.assert_called_once()
        mock_load_sample_data.assert_called_once_with(start, end, ['source1'])
        mock_st.error.assert_called_with("데이터를 가져오는 중 오류가 발생했습니다: DB Error")
        print("Finished test_get_data_db_exception")

    @patch('src.app.dashboard.st') # Mock streamlit for setup_page
    @patch.object(Dashboard, '_load_config', return_value={}) # Mock _load_config directly
    def test_get_data_empty_result_loads_sample(self, mock_load_config, mock_st):
        print("Running test_get_data_empty_result_loads_sample")
        dashboard = Dashboard(config_path=self.dummy_config_path)
        dashboard.db_manager = MagicMock()
        
        # Simulate DB returning an empty DataFrame
        empty_df = pd.DataFrame()
        dashboard.db_manager.read_sql_query.return_value = empty_df
        
        # Mock load_sample_data to see if it's called
        mock_sample_df_content = {'sample_col': [1, 2]}
        dashboard.load_sample_data = MagicMock(return_value=pd.DataFrame(mock_sample_df_content))
        
        start = date(2023, 1, 1)
        end = date(2023, 1, 31)
        
        df_result = dashboard.get_data(start, end, ['source1'])
        
        self.assertTrue(df_result.equals(pd.DataFrame(mock_sample_df_content)))
        # In this specific case (empty df from db), using_sample_data is NOT set to True by get_data itself.
        # The load_sample_data is called, but the flag is for exceptions during DB read.
        # This behavior could be debated, but current code in dashboard.py only sets the flag on Exception.
        # If the requirement is to set it also for empty df, dashboard.py needs change.
        # For now, testing current behavior:
        self.assertFalse(dashboard.using_sample_data) 
        dashboard.db_manager.read_sql_query.assert_called_once()
        dashboard.load_sample_data.assert_called_once_with(start, end, ['source1'])
        print("Finished test_get_data_empty_result_loads_sample")

    @patch('src.app.dashboard.st') # Mock streamlit for setup_page
    @patch.object(Dashboard, '_load_config', return_value={}) # Mock _load_config
    def test_load_sample_data(self, mock_load_config, mock_st):
        print("Running test_load_sample_data")
        # No need to mock DatabaseManager as load_sample_data is standalone
        dashboard = Dashboard(config_path=self.dummy_config_path)
        
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 5) # Short range for manageable sample data
        sources = ['Test Source 1', 'Test Source 2']
        
        sample_df = dashboard.load_sample_data(start_date, end_date, sources)
        
        self.assertIsInstance(sample_df, pd.DataFrame)
        self.assertFalse(sample_df.empty)
        
        expected_columns = [
            'date', 'source', 'campaign', 'impressions', 'clicks', 'cost', 
            'conversions', 'ctr', 'conversion_rate', 'cost_per_click', 
            'cost_per_conversion'
        ]
        for col in expected_columns:
            self.assertIn(col, sample_df.columns)
            
        # Check if dates are within range
        sample_df['date'] = pd.to_datetime(sample_df['date']).dt.date
        self.assertTrue(sample_df['date'].min() >= start_date)
        self.assertTrue(sample_df['date'].max() <= end_date)
        
        # Check if sources are correctly represented
        self.assertListEqual(sorted(sample_df['source'].unique()), sorted(sources))
        print("Finished test_load_sample_data")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
