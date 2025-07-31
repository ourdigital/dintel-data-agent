# Data Analysis Agent

A comprehensive data analysis automation solution for collecting, processing, analyzing, and visualizing data from multiple sources including web analytics, advertising platforms, and CSV files.

## Overview

This project focuses on automating repetitive data analysis tasks and improving analysis efficiency. It consists of four main layers:

1. **Data Collection Layer**: Collect data from various sources and integrate into a central data repository
2. **Data Preparation Layer**: Clean and prepare collected data for analysis
3. **Data Analysis Layer**: Perform exploratory data analysis (EDA) and in-depth analysis using prepared data
4. **Reporting Layer**: Visualize analysis results and generate reports for stakeholder sharing

## Supported Data Sources

- Google Analytics
- Google Search Console
- Meta Ads
- YouTube Analytics
- Google Ads
- Naver/Kakao Ads CSV
- Other CSV/Excel files

## Installation

```bash
# Clone repository
git clone git@github.com:youruser/data-analysis-agent.git
cd data-analysis-agent

# Create and activate virtual environment (Option 1: using venv)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create and activate virtual environment (Option 2: using conda)
conda create -n data-analysis-agent python=3.12 -y
conda activate data-analysis-agent

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Configuration

1. Copy `config/api_credentials.yaml.example` to `config/api_credentials.yaml` and enter required API credentials.
2. Adjust data pipeline settings in `config/pipeline_config.yaml`.
3. Copy `.env.example` to `.env` and set environment variables for database and API authentication.

## Usage

### Data Collection and Processing

```bash
# Collect data from all sources
python main.py --action collect-all

# Collect data from specific source
python main.py --action collect --source google_analytics

# Process data
python main.py --action process
```

### Analysis and Visualization

Use Jupyter Notebook for exploratory data analysis:

```bash
jupyter lab notebooks/eda_template.ipynb
```

Or run the Streamlit dashboard:

```bash
streamlit run src/app/dashboard.py
```

## Project Structure

- `src/data/`: Data collection and processing modules
- `src/database/`: Database management modules
- `src/analysis/`: Data analysis modules
- `src/visualization/`: Data visualization modules
- `src/app/`: Streamlit dashboard application
- `notebooks/`: Jupyter notebooks
- `tests/`: Unit tests

## Technology Stack

- **Primary Language**: Python
- **Database**: MySQL, SQLite
- **Data Processing/Analysis**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Analysis Environment**: Jupyter Notebook/Lab, Google Colab
- **Web App/Reporting**: Streamlit

## License

MIT

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request