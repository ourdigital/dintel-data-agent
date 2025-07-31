#!/usr/bin/env python
"""
Data Analysis Agent Main Script.
Collects, processes, analyzes data from various sources and reports results.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import subprocess

# Import internal modules
from src.utils.logging_config import setup_logging
from src.database.db_manager import DatabaseManager
from src.data.acquisition import DataAcquisition
from src.data.processing import DataProcessor
import src.visualization.plotting as plotting

# Setup logging
logger = setup_logging()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Analysis Agent")
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['collect', 'collect-all', 'process', 'analyze', 'visualize', 'dashboard', 'pipeline'],
        default='pipeline',
        help='Action to execute'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Data source (required when action is collect)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline_config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--credentials',
        type=str,
        default='config/api_credentials.yaml',
        help='Credentials file path'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()

def collect_data(args) -> Dict[str, pd.DataFrame]:
    """
    Collect data from specified source or all sources.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Collected data
    """
    logger.info("Starting data collection")
    
    # Create data acquisition object
    acquisition = DataAcquisition(
        credentials_path=args.credentials,
        config_path=args.config
    )
    
    # Collect data from single source or all sources
    if args.action == 'collect' and args.source:
        logger.info(f"Collecting data from source '{args.source}'...")
        source_data = acquisition.collect_data_from_source(args.source)
        
        if source_data.empty:
            logger.warning(f"Cannot collect data from source '{args.source}'.")
            return {}
        
        # Save to CSV and DB
        acquisition.save_to_csv(source_data, args.source)
        table_name = f"{args.source.lower()}_data"
        acquisition.save_to_database(source_data, table_name)
        
        return {args.source: source_data}
    
    else:  # All sources
        logger.info("Collecting data from all enabled sources...")
        return acquisition.run_collection_pipeline()

def process_data(args, collected_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Process collected data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    collected_data : Dict[str, pd.DataFrame], optional
        Already collected data. If None, fetch from DB.
        
    Returns
    -------
    pd.DataFrame
        Processed data
    """
    logger.info("Starting data processing")
    
    # Create data processor object
    processor = DataProcessor(config_path=args.config)
    
    # Create database manager
    db_manager = DatabaseManager(config_path=args.config)
    
    # If no collected data, fetch from DB
    if not collected_data:
        logger.info("Fetching raw data from database")
        collected_data = {}
        
        with db_manager:
            # Get table list
            tables = db_manager.execute_query_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_data'"
            )
            
            # Fetch data from each table
            for table in tables:
                table_name = table[0]
                source_name = table_name.replace('_data', '')
                
                try:
                    data = db_manager.read_sql_table(table_name)
                    if not data.empty:
                        collected_data[source_name] = data
                        logger.info(f"Loaded {len(data)} rows from table '{table_name}'")
                except Exception as e:
                    logger.error(f"Failed to load data from table '{table_name}': {e}")
    
    # Return empty DataFrame if no collected data
    if not collected_data:
        logger.warning("No data to process.")
        return pd.DataFrame()
    
    # Merge all DataFrames
    if len(collected_data) > 1:
        merged_data = processor.merge_dataframes(collected_data)
    else:
        source_name, df = next(iter(collected_data.items()))
        merged_data = df.copy()
    
    # Execute data processing pipeline
    processed_data = processor.process_pipeline(merged_data)
    
    # Save processed data
    processor.save_processed_data(processed_data, f"{args.output_dir}/processed/")
    
    # Save processed data to DB
    with db_manager:
        db_manager.dataframe_to_sql(processed_data, "processed_data", if_exists='replace')
    
    logger.info(f"Data processing completed, processed rows: {len(processed_data)}")
    return processed_data

def analyze_data(args, processed_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Analyze processed data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    processed_data : pd.DataFrame, optional
        Processed data. If None, fetch from DB.
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    logger.info("Starting data analysis")
    
    # Create database manager
    db_manager = DatabaseManager(config_path=args.config)
    
    # If no processed data, fetch from DB
    if processed_data is None or processed_data.empty:
        logger.info("Fetching processed data from database")
        
        with db_manager:
            try:
                processed_data = db_manager.read_sql_table("processed_data")
                logger.info(f"Processed data loaded, rows: {len(processed_data)}")
            except Exception as e:
                logger.error(f"Failed to load processed data: {e}")
                return {}
    
    # Return empty result if no data
    if processed_data.empty:
        logger.warning("No data to analyze.")
        return {}
    
    # Create directory for saving results
    os.makedirs(f"{args.output_dir}/analysis", exist_ok=True)
    
    # Analysis results dictionary
    analysis_results = {}
    
    # Calculate basic statistics
    try:
        stats = processed_data.describe().to_dict()
        analysis_results['basic_stats'] = stats
        
        # Save statistics results to CSV
        stats_df = processed_data.describe().reset_index()
        stats_df.to_csv(f"{args.output_dir}/analysis/basic_stats.csv")
        logger.info("Basic statistics calculation completed")
    except Exception as e:
        logger.error(f"Failed to calculate basic statistics: {e}")
    
    # Summary by source
    try:
        if 'source' in processed_data.columns:
            source_summary = processed_data.groupby('source').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).to_dict()
            
            analysis_results['source_summary'] = source_summary
            
            # Save source summary to CSV
            source_summary_df = processed_data.groupby('source').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            source_summary_df.to_csv(f"{args.output_dir}/analysis/source_summary.csv", index=False)
            logger.info("Source summary calculation completed")
    except Exception as e:
        logger.error(f"Failed to calculate source summary: {e}")
    
    # Time series analysis
    try:
        if 'date' in processed_data.columns:
            # Check and convert date format
            if processed_data['date'].dtype != 'datetime64[ns]':
                processed_data['date'] = pd.to_datetime(processed_data['date'])
            
            # Daily aggregation
            daily_data = processed_data.groupby('date').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            analysis_results['time_series'] = daily_data.to_dict('records')
            
            # Save time series data to CSV
            daily_data.to_csv(f"{args.output_dir}/analysis/daily_metrics.csv", index=False)
            logger.info("Time series analysis completed")
    except Exception as e:
        logger.error(f"Failed to perform time series analysis: {e}")
    
    # Correlation analysis
    try:
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        corr_matrix = processed_data[numeric_cols].corr().to_dict()
        
        analysis_results['correlation'] = corr_matrix
        
        # Save correlation matrix to CSV
        corr_df = processed_data[numeric_cols].corr().reset_index()
        corr_df.to_csv(f"{args.output_dir}/analysis/correlation_matrix.csv", index=False)
        logger.info("Correlation analysis completed")
    except Exception as e:
        logger.error(f"Failed to perform correlation analysis: {e}")
    
    logger.info("Data analysis completed")
    return analysis_results

def visualize_data(args, processed_data: Optional[pd.DataFrame] = None, 
                  analysis_results: Optional[Dict[str, Any]] = None) -> bool:
    """
    Visualize processed data and analysis results.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    processed_data : pd.DataFrame, optional
        Processed data. If None, fetch from DB.
    analysis_results : Dict[str, Any], optional
        Analysis results. If None, fetch from CSV files generated in analysis step.
        
    Returns
    -------
    bool
        Success status
    """
    logger.info("Starting data visualization")
    
    # Create database manager
    db_manager = DatabaseManager(config_path=args.config)
    
    # If no processed data, fetch from DB
    if processed_data is None or processed_data.empty:
        logger.info("Fetching processed data from database")
        
        with db_manager:
            try:
                processed_data = db_manager.read_sql_table("processed_data")
                logger.info(f"Processed data loaded, rows: {len(processed_data)}")
            except Exception as e:
                logger.error(f"Failed to load processed data: {e}")
                return False
    
    # Return false if no data
    if processed_data.empty:
        logger.warning("No data to visualize.")
        return False
    
    # Create directory for saving visualization results
    viz_dir = f"{args.output_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # Check and convert date format
        if 'date' in processed_data.columns and processed_data['date'].dtype != 'datetime64[ns]':
            processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # 1. Correlation heatmap
        try:
            numeric_cols = processed_data.select_dtypes(include=['number']).columns
            fig1, ax1 = plotting.create_correlation_heatmap(
                processed_data,
                columns=numeric_cols,
                save_path=f"{viz_dir}/correlation_heatmap.png"
            )
            logger.info("Correlation heatmap creation completed")
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {e}")
        
        # 2. Time series graph (daily metrics)
        if 'date' in processed_data.columns:
            try:
                # Daily data aggregation
                daily_data = processed_data.groupby('date').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # Sort by date
                daily_data = daily_data.sort_values('date')
                
                # Create time series graph
                fig2, ax2 = plotting.create_time_series_plot(
                    daily_data,
                    date_column='date',
                    value_columns=['clicks', 'conversions'],
                    title='Daily clicks and conversions trend',
                    figsize=(12, 6),
                    save_path=f"{viz_dir}/daily_metrics.png"
                )
                logger.info("Time series graph creation completed")
                
                # Interactive time series graph
                fig_interactive = plotting.create_interactive_plot(
                    daily_data,
                    plot_type='line',
                    x='date',
                    y=['impressions', 'clicks', 'conversions', 'cost'],
                    title='Daily metrics trend (interactive)'
                )
                
                plotting.save_plot_to_html(
                    fig_interactive,
                    f"{viz_dir}/daily_metrics_interactive.html"
                )
                logger.info("Interactive time series graph creation completed")
            except Exception as e:
                logger.error(f"Failed to create time series graph: {e}")
        
        # 3. Source performance bar chart
        if 'source' in processed_data.columns:
            try:
                # Source data aggregation
                source_data = processed_data.groupby('source').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # Create source bar chart
                fig3, ax3 = plotting.create_bar_chart(
                    source_data,
                    category_column='source',
                    value_column='conversions',
                    title='Conversions by source',
                    figsize=(10, 6),
                    save_path=f"{viz_dir}/source_conversions.png"
                )
                logger.info("Source bar chart creation completed")
                
                # Source cost bar chart
                fig4, ax4 = plotting.create_bar_chart(
                    source_data,
                    category_column='source',
                    value_column='cost',
                    title='Cost by source',
                    figsize=(10, 6),
                    save_path=f"{viz_dir}/source_cost.png"
                )
                logger.info("Source cost bar chart creation completed")
            except Exception as e:
                logger.error(f"Failed to create source charts: {e}")
        
        # 4. Cost vs conversions scatter plot
        try:
            # Campaign or source data
            scatter_data = processed_data.copy()
            
            if 'campaign' in processed_data.columns:
                group_by = 'campaign'
            elif 'source' in processed_data.columns:
                group_by = 'source'
            else:
                group_by = None
            
            if group_by:
                scatter_data = processed_data.groupby(group_by).agg({
                    'clicks': 'sum',
                    'conversions': 'sum',
                    'cost': 'sum'
                }).reset_index()
                
                # Create scatter plot
                fig5, ax5 = plotting.create_scatter_plot(
                    scatter_data,
                    x_column='cost',
                    y_column='conversions',
                    size_column='clicks',
                    title=f'Cost vs conversions by {group_by} (size: clicks)',
                    add_trendline=True,
                    save_path=f"{viz_dir}/cost_vs_conversions.png"
                )
                logger.info("Cost vs conversions scatter plot creation completed")
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
        
        # 5. Pie chart (source or campaign ratio)
        try:
            if 'source' in processed_data.columns:
                # Source conversion ratio
                fig6, ax6 = plotting.create_pie_chart(
                    processed_data,
                    category_column='source',
                    value_column='conversions',
                    title='Conversion ratio by source',
                    save_path=f"{viz_dir}/source_conversion_pie.png"
                )
                logger.info("Source conversion ratio pie chart creation completed")
            
            if 'campaign' in processed_data.columns:
                # Campaign cost ratio (top 5)
                campaign_cost = processed_data.groupby('campaign')['cost'].sum().reset_index()
                campaign_cost = campaign_cost.sort_values('cost', ascending=False).head(5)
                
                fig7, ax7 = plotting.create_pie_chart(
                    campaign_cost,
                    category_column='campaign',
                    value_column='cost',
                    title='Cost ratio by top 5 campaigns',
                    save_path=f"{viz_dir}/campaign_cost_pie.png"
                )
                logger.info("Campaign cost ratio pie chart creation completed")
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
        
        logger.info(f"Data visualization completed. Results saved to '{viz_dir}' directory.")
        return True
        
    except Exception as e:
        logger.error(f"Error occurred during data visualization: {e}")
        return False

def run_dashboard(args):
    """
    Run Streamlit dashboard.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Running Streamlit dashboard...")
    
    try:
        dashboard_path = "src/app/dashboard.py"
        
        # Run Streamlit
        cmd = ["streamlit", "run", dashboard_path, "--server.port=8501"]
        
        if args.debug:
            cmd.append("--logger.level=debug")
        
        logger.info(f"Executing dashboard command: {' '.join(cmd)}")
        
        # Start Streamlit process
        process = subprocess.Popen(cmd)
        
        logger.info("Dashboard started. Access http://localhost:8501 in your browser.")
        logger.info("Press Ctrl+C to exit.")
        
        # Wait until process terminates
        process.wait()
        
    except Exception as e:
        logger.error(f"Error occurred while running dashboard: {e}")
        return False
    
    return True

def run_pipeline(args):
    """
    Run complete data pipeline.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    bool
        Success status
    """
    logger.info("Starting complete data pipeline execution")
    
    try:
        # 1. Data collection
        collected_data = collect_data(args)
        
        if not collected_data:
            logger.warning("No collected data or collection failed.")
            return False
        
        # 2. Data processing
        processed_data = process_data(args, collected_data)
        
        if processed_data.empty:
            logger.warning("No processed data or processing failed.")
            return False
        
        # 3. Data analysis
        analysis_results = analyze_data(args, processed_data)
        
        # 4. Data visualization
        visualize_data(args, processed_data, analysis_results)
        
        logger.info("Complete data pipeline execution completed")
        
        # 5. Check dashboard execution
        run_dashboard_prompt = input("Would you like to run Streamlit dashboard? (y/n): ").strip().lower()
        
        if run_dashboard_prompt == 'y':
            run_dashboard(args)
        
        return True
        
    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {e}")
        return False

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode activated.")
    
    # Execute selected action
    try:
        if args.action == 'collect' or args.action == 'collect-all':
            collect_data(args)
        elif args.action == 'process':
            process_data(args)
        elif args.action == 'analyze':
            analyze_data(args)
        elif args.action == 'visualize':
            visualize_data(args)
        elif args.action == 'dashboard':
            run_dashboard(args)
        elif args.action == 'pipeline':
            run_pipeline(args)
        else:
            logger.error(f"Unknown action: {args.action}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Return exit code
    sys.exit(main())