# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running the Application
```bash
# Execute full data pipeline (collect, process, analyze, visualize)
python main.py --action pipeline

# Run specific operations
python main.py --action collect-all                    # Collect from all sources
python main.py --action collect --source google_ads    # Collect from specific source
python main.py --action process                        # Process collected data
python main.py --action analyze                        # Analyze processed data
python main.py --action visualize                      # Generate visualizations
python main.py --action dashboard                      # Launch Streamlit dashboard

# Debug mode
python main.py --action pipeline --debug
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_acquisition.py
python -m pytest tests/test_processing.py
python -m pytest tests/test_visualization.py
```

### Dashboard
```bash
# Launch Streamlit dashboard
streamlit run src/app/dashboard.py

# Dashboard with custom port
streamlit run src/app/dashboard.py --server.port=8502
```

### Jupyter Notebooks
```bash
# Start Jupyter Lab for exploratory analysis
jupyter lab notebooks/eda_template.ipynb
```

## Code Architecture

### Core Components

**Data Collection Layer** (`src/data/connectors/`)
- **google_ads.py**: Google Ads API integration for campaign performance data
- **google_analytics.py**: Google Analytics 4 API for website traffic metrics
- **meta_ads.py**: Meta Business API for Facebook/Instagram advertising data
- **naver_ads.py**: Naver advertising platform data (CSV-based)
- **kakao_ads.py**: Kakao advertising platform data (CSV-based)

**Data Processing** (`src/data/`)
- **acquisition.py**: Orchestrates data collection from multiple sources
- **processing.py**: Data cleaning, transformation, and derived metric calculation

**Database Layer** (`src/database/`)
- **db_manager.py**: Unified SQLite/MySQL interface with context manager support
- **models.py**: Database schema definitions and table models

**Analysis Layer** (`src/analysis/`)
- **statistics.py**: Descriptive and inferential statistical analysis
- **modeling.py**: Machine learning and predictive modeling capabilities

**Visualization Layer** (`src/visualization/`)
- **plotting.py**: Chart generation (matplotlib, seaborn, plotly) with export capabilities

**Application Layer** (`src/app/`)
- **dashboard.py**: Multi-page Streamlit dashboard for data visualization
- **components/**: Reusable UI components for the dashboard

### Data Flow Architecture

1. **Collection**: External APIs → Connectors → Raw CSV/Database storage
2. **Processing**: Raw data → Cleaning → Transformation → Derived metrics
3. **Analysis**: Processed data → Statistical analysis → Insights generation
4. **Visualization**: Analysis results → Charts → Dashboard/Export

### Configuration Management

**Pipeline Configuration** (`config/pipeline_config.yaml`)
- Database settings (SQLite/MySQL)
- Data source configurations and schedules
- Processing parameters and metric definitions
- Visualization preferences and dashboard settings

**API Credentials** (`config/api_credentials.yaml.example`)
- Google Analytics/Ads API credentials
- Meta Business API tokens
- Database connection strings
- Copy to `config/api_credentials.yaml` and populate with actual credentials

### Key Design Patterns

**Modular Connector Architecture**: Each data source implements consistent interface patterns, making it easy to add new sources without modifying core logic.

**Configuration-Driven Design**: Centralized YAML configuration allows flexible deployment across different environments and use cases.

**Context Manager Pattern**: Database connections use context managers for proper resource cleanup and error handling.

**Pipeline Architecture**: Sequential data processing stages with clear input/output contracts between components.

## Common Development Tasks

### Adding a New Data Source

1. Create new connector in `src/data/connectors/new_source.py`
2. Implement required methods: `collect_data()`, `validate_credentials()`, `get_schema()`
3. Add configuration section to `config/pipeline_config.yaml`
4. Update `src/data/acquisition.py` to include new source
5. Write tests in `tests/test_new_source.py`

### Extending Analysis Capabilities

1. Add new analysis functions to `src/analysis/statistics.py` or `src/analysis/modeling.py`
2. Update the analysis pipeline in `main.py` to include new calculations
3. Create corresponding visualizations in `src/visualization/plotting.py`
4. Add dashboard components in `src/app/dashboard.py`

### Database Schema Changes

1. Update models in `src/database/models.py`
2. Modify table creation logic in `src/database/db_manager.py`
3. Test migration scripts for existing data
4. Update data processing pipeline to handle new schema

## Key Configuration Files

- `config/pipeline_config.yaml`: Main configuration for all pipeline components
- `config/api_credentials.yaml`: API keys and database credentials (not in git)
- `requirements.txt`: Python dependencies
- `setup.py`: Package configuration and console scripts

## Output Directories

- `data/output/processed/`: Processed data files (CSV format)
- `data/output/visualizations/`: Generated charts and graphs
- `data/output/analysis/`: Statistical analysis results
- `logs/`: Application logs and debug information

## Data Sources Supported

- Google Analytics 4 (sessions, pageviews, bounce rate, conversion tracking)
- Google Ads (campaign performance, keywords, ad groups)
- Meta Ads (Facebook/Instagram campaigns, audience insights)
- Naver Ads (Korean market advertising data via CSV)
- Kakao Ads (Korean market advertising data via CSV)
- CSV/Excel files (custom data import)

## Dashboard Features

- Multi-page Streamlit interface with navigation
- Interactive filtering by date range, data source, and metrics
- Real-time data updates from database
- Export capabilities for charts and data
- Custom report generation and scheduling