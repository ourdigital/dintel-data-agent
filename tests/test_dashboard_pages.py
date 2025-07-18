import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import page rendering functions
from src.app.pages import overview, traffic_analysis, campaign_performance, conversion_analysis, custom_reports

class TestDashboardPages(unittest.TestCase):

    def setUp(self):
        # Create a mock Dashboard instance
        self.mock_dashboard = MagicMock()
        
        # Sample DataFrame to be returned by mock_dashboard.get_data
        self.sample_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'impressions': [1000, 1200, 1100],
            'clicks': [100, 150, 120],
            'conversions': [10, 12, 8],
            'cost': [50.0, 60.0, 55.0],
            'source': ['Google Ads', 'Meta Ads', 'Google Ads'],
            'campaign': ['Campaign A', 'Campaign B', 'Campaign A'],
            'device': ['Desktop', 'Mobile', 'Desktop'],
            'conversion_type': ['Purchase', 'Lead', 'Purchase'],
            'conversion_value': [1200.0, 50.0, 1000.0]
        })
        # Sample for numeric only df for correlation
        self.numeric_sample_df = pd.DataFrame({
            'impressions': [1000, 1200, 1100],
            'clicks': [100, 150, 120],
            'conversions': [10, 12, 8],
            'cost': [50.0, 60.0, 55.0],
        })


        self.mock_dashboard.get_data.return_value = self.sample_df
        
        # Mock config (can be expanded if pages use more specific config)
        self.mock_dashboard.config = {'app': {}} 
        
        # Dummy arguments for render_page functions
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 31)
        self.sources = ['Google Ads', 'Meta Ads']
        self.metrics = ['impressions', 'clicks', 'conversions', 'cost']

    @patch('src.app.pages.overview.st') # Patch streamlit in the context of the overview module
    @patch('src.app.pages.overview.px') # Patch plotly.express
    @patch('src.app.pages.overview.plt') # Patch matplotlib.pyplot
    @patch('src.app.pages.overview.sns') # Patch seaborn
    def test_render_overview_page(self, mock_sns, mock_plt, mock_px, mock_st):
        print("Running test_render_overview_page")
        try:
            # Ensure get_data returns a df with necessary numeric columns for correlation
            self.mock_dashboard.get_data.return_value = self.sample_df.copy()
            overview.render_page(
                self.mock_dashboard, self.start_date, self.end_date, 
                self.sources, self.metrics
            )
            # Basic assertions: check if some streamlit methods were called
            mock_st.header.assert_called_with('개요')
            mock_st.write.assert_called()
            mock_st.columns.assert_called()
            mock_st.metric.assert_called()
            mock_st.subheader.assert_called()
            mock_st.selectbox.assert_called()
            mock_st.plotly_chart.assert_called()
            mock_st.pyplot.assert_called() # For the seaborn heatmap
            mock_px.line.assert_called()
            mock_px.bar.assert_called()
            mock_sns.heatmap.assert_called()

        except Exception as e:
            self.fail(f"render_overview_page raised an exception: {e}")
        print("Finished test_render_overview_page")

    @patch('src.app.pages.traffic_analysis.st')
    @patch('src.app.pages.traffic_analysis.px')
    @patch('src.app.pages.traffic_analysis.go')
    def test_render_traffic_analysis_page(self, mock_go, mock_px, mock_st):
        print("Running test_render_traffic_analysis_page")
        try:
            traffic_analysis.render_page(
                self.mock_dashboard, self.start_date, self.end_date, self.sources
            )
            mock_st.header.assert_called_with('트래픽 분석')
            mock_st.plotly_chart.assert_called()
            mock_px.pie.assert_called()
            mock_go.Figure.assert_called() # For combined chart
        except Exception as e:
            self.fail(f"render_traffic_analysis_page raised an exception: {e}")
        print("Finished test_render_traffic_analysis_page")

    @patch('src.app.pages.campaign_performance.st')
    @patch('src.app.pages.campaign_performance.px')
    def test_render_campaign_performance_page(self, mock_px, mock_st):
        print("Running test_render_campaign_performance_page")
        try:
            campaign_performance.render_page(
                self.mock_dashboard, self.start_date, self.end_date, 
                self.sources, self.metrics
            )
            mock_st.header.assert_called_with('캠페인 성과 분석')
            mock_st.plotly_chart.assert_called()
            mock_px.bar.assert_called()
            mock_px.scatter.assert_called()
        except Exception as e:
            self.fail(f"render_campaign_performance_page raised an exception: {e}")
        print("Finished test_render_campaign_performance_page")

    @patch('src.app.pages.conversion_analysis.st')
    @patch('src.app.pages.conversion_analysis.px')
    @patch('src.app.pages.conversion_analysis.go')
    def test_render_conversion_analysis_page(self, mock_go, mock_px, mock_st):
        print("Running test_render_conversion_analysis_page")
        try:
            conversion_analysis.render_page(
                self.mock_dashboard, self.start_date, self.end_date, self.sources
            )
            mock_st.header.assert_called_with('전환 분석')
            mock_st.plotly_chart.assert_called()
            mock_px.pie.assert_called()
            mock_px.bar.assert_called()
            mock_px.scatter.assert_called()
            mock_go.Figure.assert_called()
        except Exception as e:
            self.fail(f"render_conversion_analysis_page raised an exception: {e}")
        print("Finished test_render_conversion_analysis_page")

    @patch('src.app.pages.custom_reports.st')
    @patch('src.app.pages.custom_reports.px')
    def test_render_custom_reports_page(self, mock_px, mock_st):
        print("Running test_render_custom_reports_page")
        
        # Mock user selections for different report types
        report_types_configs = [
            {'type': '시계열 분석', 'x': 'date', 'y': self.metrics, 'group': 'source'},
            {'type': '비교 분석', 'x': 'source', 'y': self.metrics[0], 'chart': '막대 그래프'},
            {'type': '상관관계 분석', 'x': 'clicks', 'y': 'conversions', 'group': 'source'},
            {'type': '데이터 테이블', 'table_type': '요약 테이블', 'group_by': ['source'], 'agg': self.metrics},
            {'type': '데이터 테이블', 'table_type': '피벗 테이블', 'rows': 'source', 'cols': 'campaign', 'values': self.metrics[0], 'aggfunc': '합계'}
        ]

        for config in report_types_configs:
            with self.subTest(report_type=config['type']):
                # Reset mocks for each subtest if necessary, or ensure calls are specific enough
                mock_st.reset_mock()
                mock_px.reset_mock()

                # Simulate streamlit widget return values
                # The key is to make st.selectbox/multiselect return appropriate values for each subtest
                
                # Default return for all selectbox/multiselect calls
                mock_st.selectbox.return_value = config['type'] # Initial report type selection
                mock_st.multiselect.return_value = self.metrics # Default for metrics

                if config['type'] == '시계열 분석':
                    mock_st.selectbox.side_effect = [
                        config['type'], # report_type
                        config['x'],    # x_axis
                        config['group'] # groupby
                    ]
                    mock_st.multiselect.return_value = config['y'] # y_axis
                elif config['type'] == '비교 분석':
                     mock_st.selectbox.side_effect = [
                        config['type'], # report_type
                        config['x'],    # x_axis
                        config['y']     # y_axis
                    ]
                     mock_st.radio.side_effect = ['값 기준', config['chart']] # sort_by, chart_type
                elif config['type'] == '상관관계 분석':
                    mock_st.selectbox.side_effect = [
                        config['type'], # report_type
                        config['x'],    # x_axis for correlation
                        config['y'],    # y_axis for correlation
                        config['group'],# groupby for correlation
                        '없음'          # size_by for correlation
                    ]
                elif config['type'] == '데이터 테이블':
                    mock_st.radio.return_value = config['table_type'] # table_type
                    if config['table_type'] == '요약 테이블':
                        mock_st.multiselect.side_effect = [
                            config['group_by'], # groupby_cols
                            config['agg']       # agg_metrics
                        ]
                    else: # 피벗 테이블
                        mock_st.selectbox.side_effect = [
                            config['type'],     # report_type
                            config['rows'],     # rows
                            config['cols'],     # columns
                            config['values'],   # values
                            config['aggfunc']   # aggfunc
                        ]
                
                try:
                    custom_reports.render_page(
                        self.mock_dashboard, self.start_date, self.end_date, 
                        self.sources, self.metrics
                    )
                    mock_st.header.assert_called_with('사용자 정의 보고서')
                    # More specific assertions can be added here based on report type
                    if config['type'] != '데이터 테이블': # Plotly chart is not always called for data table
                         mock_st.plotly_chart.assert_called()

                except Exception as e:
                    self.fail(f"render_custom_reports_page (type: {config['type']}) raised an exception: {e}")
        print("Finished test_render_custom_reports_page")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
