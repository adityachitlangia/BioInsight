# BioInsight: Bioprocess Analytics Dashboard

## Project Overview
BioInsight is a data analytics and machine learning platform designed to analyze, visualize, and predict bioprocess outcomes. This project focuses on analyzing bioprocess data across different scales (1 mL and 30 L) to predict key performance indicators such as Final OD and GFPuv production.

## Features
- **Data Processing & Visualization**: Intuitive interface for exploring process data with interactive time series visualization
- **Feature Selection**: Advanced analysis of important features with correlation matrices and distribution plots
- **Model Results**: Comparison of multiple machine learning models (Random Forest, XGBoost, SVR, PLS) with performance metrics
- **Interactive Predictions**: Make real-time predictions by adjusting feature values
- **PLS Component Analysis**: Visualize PLS components and explained variance

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Interactive dashboard framework
- **Scikit-learn**: Machine learning model development
- **XGBoost**: Gradient boosting implementation
- **Plotly**: Interactive data visualization
- **Pandas/NumPy**: Data manipulation and numerical operations

## Project Structure
```
├── data/
│   └── Wrangled_Combined_Batch_Dataset.xlsx  # Main dataset
├── models/                                   # Trained model files
├── src/
│   ├── dashboard/
│   │   └── app.py                            # Streamlit dashboard application
│   └── model_development.py                  # Model training pipeline
└── README.md                                 # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/bioinsight.git
cd bioinsight
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard (You will need to upload Wrangled_Combined_Batch_Dataset.xlsx under data directory)
To launch the BioInsight dashboard:
```bash
cd src/dashboard 
streamlit run app.py
```

The dashboard will be accessible in your web browser at http://localhost:8501

### Training Models
To train or retrain the machine learning models:
```bash
python src/model_development.py
```

## Dashboard Sections

### 1. Data Processing
- View and analyze raw process data
- Visualize missing values
- Explore time series data by batch

### 2. Feature Selection
- Examine feature importance rankings
- Analyze feature correlations
- Explore feature distributions with statistical insights

### 3. Model Results
- Compare model performance (RMSE, R² score)
- Visualize actual vs predicted values
- Analyze PLS components and explained variance
- Make interactive predictions

## Contributors
- Aditya Chitlangia
