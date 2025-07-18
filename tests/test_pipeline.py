import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
# This is necessary for the tests to find the 'src' module
# Assuming 'tests' directory is at the same level as 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import pipeline # Import the module to be tested

class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Common setup for arguments, can be overridden in specific tests
        self.mock_args = argparse.Namespace(
            action='pipeline',
            source=None,
            output_dir='data/output',
            config='config/pipeline_config.yaml',
            credentials='config/api_credentials.yaml',
            debug=False
        )

    @patch('src.pipeline.DataAcquisition')
    @patch('src.pipeline.DatabaseManager') # Mock DatabaseManager if it's used directly in collect_data for saving
    def test_collect_data_single_source_success(self, MockDatabaseManager, MockDataAcquisition):
        print("Running test_collect_data_single_source_success")
        # Setup
        self.mock_args.action = 'collect'
        self.mock_args.source = 'test_source'
        
        mock_acquisition_instance = MockDataAcquisition.return_value
        mock_df = pd.DataFrame({'data': ['sample']})
        mock_acquisition_instance.collect_data_from_source.return_value = mock_df
        
        # Mock for DatabaseManager if saving happens within collect_data
        mock_db_manager_instance = MockDatabaseManager.return_value

        # Execute
        result = pipeline.collect_data(self.mock_args)

        # Assert
        MockDataAcquisition.assert_called_once_with(
            credentials_path=self.mock_args.credentials,
            config_path=self.mock_args.config
        )
        mock_acquisition_instance.collect_data_from_source.assert_called_once_with('test_source')
        mock_acquisition_instance.save_to_csv.assert_called_once_with(mock_df, 'test_source')
        mock_acquisition_instance.save_to_database.assert_called_once_with(mock_df, 'test_source_data')
        self.assertEqual(result, {'test_source': mock_df})
        print("Finished test_collect_data_single_source_success")

    @patch('src.pipeline.DataAcquisition')
    def test_collect_data_all_sources(self, MockDataAcquisition):
        print("Running test_collect_data_all_sources")
        # Setup
        self.mock_args.action = 'collect-all' # Or any action that triggers all source collection
        
        mock_acquisition_instance = MockDataAcquisition.return_value
        mock_all_data = {'source1': pd.DataFrame({'data': [1]}), 'source2': pd.DataFrame({'data': [2]})}
        mock_acquisition_instance.run_collection_pipeline.return_value = mock_all_data

        # Execute
        result = pipeline.collect_data(self.mock_args)

        # Assert
        MockDataAcquisition.assert_called_once_with(
            credentials_path=self.mock_args.credentials,
            config_path=self.mock_args.config
        )
        mock_acquisition_instance.run_collection_pipeline.assert_called_once()
        self.assertEqual(result, mock_all_data)
        print("Finished test_collect_data_all_sources")

    @patch('src.pipeline.DataProcessor')
    @patch('src.pipeline.DatabaseManager')
    def test_process_data_with_collected_data(self, MockDatabaseManager, MockDataProcessor):
        print("Running test_process_data_with_collected_data")
        # Setup
        mock_processor_instance = MockDataProcessor.return_value
        mock_db_manager_instance = MockDatabaseManager.return_value
        
        collected_data = {'source1': pd.DataFrame({'raw': [1]})}
        processed_df = pd.DataFrame({'processed': [1]})
        mock_processor_instance.process_pipeline.return_value = processed_df
        mock_processor_instance.merge_dataframes.return_value = pd.DataFrame({'merged': [1]})


        # Execute
        result = pipeline.process_data(self.mock_args, collected_data)

        # Assert
        MockDataProcessor.assert_called_once_with(config_path=self.mock_args.config)
        # If more than one df, merge is called. If one, it's not.
        if len(collected_data) > 1:
             mock_processor_instance.merge_dataframes.assert_called_once_with(collected_data)
        else:
            mock_processor_instance.merge_dataframes.assert_not_called()

        mock_processor_instance.process_pipeline.assert_called_once() # Argument depends on merge logic
        mock_processor_instance.save_processed_data.assert_called_once_with(processed_df, f"{self.mock_args.output_dir}/processed/")
        mock_db_manager_instance.dataframe_to_sql.assert_called_once_with(processed_df, "processed_data", if_exists='replace')
        self.assertTrue(result.equals(processed_df))
        print("Finished test_process_data_with_collected_data")

    @patch('src.pipeline.DataProcessor')
    @patch('src.pipeline.DatabaseManager')
    def test_process_data_from_db(self, MockDatabaseManager, MockDataProcessor):
        print("Running test_process_data_from_db")
        # Setup
        mock_processor_instance = MockDataProcessor.return_value
        mock_db_manager_instance = MockDatabaseManager.return_value
        
        # Simulate DB returning data
        mock_db_manager_instance.execute_query_fetchall.return_value = [('table1_data',)]
        mock_db_manager_instance.read_sql_table.return_value = pd.DataFrame({'db_data': [1]})
        
        processed_df = pd.DataFrame({'processed_db': [1]})
        mock_processor_instance.process_pipeline.return_value = processed_df
        mock_processor_instance.merge_dataframes.return_value = pd.DataFrame({'merged_db': [1]})


        # Execute
        result = pipeline.process_data(self.mock_args, collected_data=None) # No collected_data

        # Assert
        MockDatabaseManager.assert_called_once_with(config_path=self.mock_args.config)
        mock_db_manager_instance.execute_query_fetchall.assert_called_once_with(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_data'"
        )
        mock_db_manager_instance.read_sql_table.assert_called_once_with('table1_data')
        
        # merge_dataframes might be called or not depending on how many tables are found
        # For this specific mock, it finds one table, so merge_dataframes might not be called
        # or called with a single-item dict. This depends on the internal logic.
        # Let's assume it handles a single DataFrame correctly without explicit merge.
        
        mock_processor_instance.process_pipeline.assert_called_once()
        mock_processor_instance.save_processed_data.assert_called_once_with(processed_df, f"{self.mock_args.output_dir}/processed/")
        mock_db_manager_instance.dataframe_to_sql.assert_called_once_with(processed_df, "processed_data", if_exists='replace')
        self.assertTrue(result.equals(processed_df))
        print("Finished test_process_data_from_db")

    @patch('src.pipeline.DatabaseManager')
    @patch('pandas.DataFrame.describe') # To mock describe calls
    @patch('os.makedirs') # To mock directory creation
    def test_analyze_data_with_processed_data(self, mock_makedirs, mock_describe, MockDatabaseManager):
        print("Running test_analyze_data_with_processed_data")
        # Setup
        mock_db_manager_instance = MockDatabaseManager.return_value
        processed_data = pd.DataFrame({
            'impressions': [100, 200], 'clicks': [10, 20], 
            'conversions': [1, 2], 'cost': [5, 10],
            'source': ['src1', 'src2'], 'date': pd.to_datetime(['2023-01-01', '2023-01-02'])
        })
        
        # Mock describe to return a dictionary-like structure
        mock_describe.return_value.to_dict.return_value = {'stat': 1}
        mock_describe.return_value.reset_index.return_value = pd.DataFrame({'stat_df': [1]})


        # Execute
        results = pipeline.analyze_data(self.mock_args, processed_data)

        # Assert
        mock_makedirs.assert_called_once_with(f"{self.mock_args.output_dir}/analysis", exist_ok=True)
        self.assertIn('basic_stats', results)
        self.assertIn('source_summary', results)
        self.assertIn('time_series', results)
        self.assertIn('correlation', results)
        # Further assertions could check for file saving if we mock to_csv
        print("Finished test_analyze_data_with_processed_data")


    @patch('src.pipeline.DatabaseManager')
    @patch('src.pipeline.plotting')
    @patch('os.makedirs')
    def test_visualize_data_with_processed_data(self, mock_makedirs, mock_plotting, MockDatabaseManager):
        print("Running test_visualize_data_with_processed_data")
        # Setup
        mock_db_manager_instance = MockDatabaseManager.return_value
        processed_data = pd.DataFrame({
            'impressions': [100, 200], 'clicks': [10, 20], 
            'conversions': [1, 2], 'cost': [5, 10],
            'source': ['src1', 'src2'], 'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'campaign': ['campA', 'campB']
        })
        analysis_results = {'some_analysis': 'data'} # Dummy analysis results

        # Execute
        success = pipeline.visualize_data(self.mock_args, processed_data, analysis_results)

        # Assert
        mock_makedirs.assert_called_once_with(f"{self.mock_args.output_dir}/visualizations", exist_ok=True)
        self.assertTrue(success)
        # Check if plotting functions were called (example for one)
        mock_plotting.create_correlation_heatmap.assert_called()
        mock_plotting.create_time_series_plot.assert_called()
        mock_plotting.create_interactive_plot.assert_called()
        mock_plotting.save_plot_to_html.assert_called()
        # Add more assertions for other plot calls if necessary
        print("Finished test_visualize_data_with_processed_data")

    @patch('src.pipeline.subprocess.Popen')
    def test_run_dashboard(self, mock_popen):
        print("Running test_run_dashboard")
        # Setup
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        self.mock_args.debug = False # Test non-debug case first

        # Execute
        result = pipeline.run_dashboard(self.mock_args)

        # Assert
        expected_cmd = ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501"]
        mock_popen.assert_called_once_with(expected_cmd)
        mock_process.wait.assert_called_once()
        self.assertTrue(result)
        print("Finished test_run_dashboard")

    @patch('src.pipeline.subprocess.Popen')
    def test_run_dashboard_debug_mode(self, mock_popen):
        print("Running test_run_dashboard_debug_mode")
        # Setup
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        self.mock_args.debug = True # Test debug case

        # Execute
        result = pipeline.run_dashboard(self.mock_args)

        # Assert
        expected_cmd_debug = ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501", "--logger.level=debug"]
        mock_popen.assert_called_once_with(expected_cmd_debug)
        mock_process.wait.assert_called_once()
        self.assertTrue(result)
        print("Finished test_run_dashboard_debug_mode")

    @patch('src.pipeline.collect_data')
    @patch('src.pipeline.process_data')
    @patch('src.pipeline.analyze_data')
    @patch('src.pipeline.visualize_data')
    @patch('src.pipeline.run_dashboard')
    @patch('builtins.input', return_value='n') # Mock input to not run dashboard
    def test_run_pipeline_success_no_dashboard(self, mock_input, mock_run_dashboard, mock_visualize, mock_analyze, mock_process, mock_collect):
        print("Running test_run_pipeline_success_no_dashboard")
        # Setup
        mock_collect.return_value = {'source': pd.DataFrame([1])} # Non-empty
        mock_process.return_value = pd.DataFrame([1]) # Non-empty
        mock_analyze.return_value = {'analysis': 'done'}
        mock_visualize.return_value = True

        # Execute
        result = pipeline.run_pipeline(self.mock_args)

        # Assert
        mock_collect.assert_called_once_with(self.mock_args)
        mock_process.assert_called_once_with(self.mock_args, mock_collect.return_value)
        mock_analyze.assert_called_once_with(self.mock_args, mock_process.return_value)
        mock_visualize.assert_called_once_with(self.mock_args, mock_process.return_value, mock_analyze.return_value)
        mock_input.assert_called_once_with("Streamlit 대시보드를 실행하시겠습니까? (y/n): ")
        mock_run_dashboard.assert_not_called() # Since input is 'n'
        self.assertTrue(result)
        print("Finished test_run_pipeline_success_no_dashboard")

    @patch('src.pipeline.collect_data')
    @patch('src.pipeline.process_data')
    @patch('src.pipeline.analyze_data')
    @patch('src.pipeline.visualize_data')
    @patch('src.pipeline.run_dashboard')
    @patch('builtins.input', return_value='y') # Mock input to run dashboard
    def test_run_pipeline_success_with_dashboard(self, mock_input, mock_run_dashboard, mock_visualize, mock_analyze, mock_process, mock_collect):
        print("Running test_run_pipeline_success_with_dashboard")
        # Setup
        mock_collect.return_value = {'source': pd.DataFrame([1])}
        mock_process.return_value = pd.DataFrame([1])
        mock_analyze.return_value = {'analysis': 'done'}
        mock_visualize.return_value = True
        mock_run_dashboard.return_value = True


        # Execute
        result = pipeline.run_pipeline(self.mock_args)

        # Assert
        mock_collect.assert_called_once_with(self.mock_args)
        mock_process.assert_called_once_with(self.mock_args, mock_collect.return_value)
        mock_analyze.assert_called_once_with(self.mock_args, mock_process.return_value)
        mock_visualize.assert_called_once_with(self.mock_args, mock_process.return_value, mock_analyze.return_value)
        mock_input.assert_called_once_with("Streamlit 대시보드를 실행하시겠습니까? (y/n): ")
        mock_run_dashboard.assert_called_once_with(self.mock_args)
        self.assertTrue(result)
        print("Finished test_run_pipeline_success_with_dashboard")

    @patch('src.pipeline.collect_data', return_value={}) # Simulate collect_data failing
    def test_run_pipeline_collect_fails(self, mock_collect):
        print("Running test_run_pipeline_collect_fails")
        result = pipeline.run_pipeline(self.mock_args)
        mock_collect.assert_called_once_with(self.mock_args)
        self.assertFalse(result)
        print("Finished test_run_pipeline_collect_fails")

    @patch('src.pipeline.collect_data', return_value={'source': pd.DataFrame([1])})
    @patch('src.pipeline.process_data', return_value=pd.DataFrame()) # Simulate process_data failing
    def test_run_pipeline_process_fails(self, mock_process, mock_collect):
        print("Running test_run_pipeline_process_fails")
        result = pipeline.run_pipeline(self.mock_args)
        mock_collect.assert_called_once_with(self.mock_args)
        mock_process.assert_called_once_with(self.mock_args, mock_collect.return_value)
        self.assertFalse(result)
        print("Finished test_run_pipeline_process_fails")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
