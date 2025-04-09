import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.process_data = None
        self.initial_conditions = None
        self.process_kpis = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data from Excel sheets"""
        try:
            # Load data from Excel sheets
            self.process_data = pd.read_excel(self.file_path, sheet_name='Process Data')
            self.initial_conditions = pd.read_excel(self.file_path, sheet_name='Initial Conditions')
            self.process_kpis = pd.read_excel(self.file_path, sheet_name='Process KPIs')
            
            # Display shapes and column names
            st.write("\nProcess Data shape:", self.process_data.shape)
            st.write("Process Data columns:", self.process_data.columns.tolist())
            
            st.write("\nInitial Conditions shape:", self.initial_conditions.shape)
            st.write("Initial Conditions columns:", self.initial_conditions.columns.tolist())
            
            st.write("\nProcess KPIs shape:", self.process_kpis.shape)
            st.write("Process KPIs columns:", self.process_kpis.columns.tolist())
            
            # Check for 'Batch ID' column
            if 'Batch ID' not in self.process_data.columns:
                raise ValueError("'Batch ID' column not found in Process Data")
            
            # If we get here, all data was loaded successfully
            return True
            
        except FileNotFoundError as e:
            st.error(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
            
    def subset_by_scale(self, scale):
        """Subset data based on batch scale"""
        if self.process_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
            
        if scale not in ['1 mL', '30 L']:
            raise ValueError("Scale must be either '1 mL' or '30 L'")
            
        return self.process_data[self.process_data['Scale'] == scale]
        
    def compute_batch_statistics(self, data):
        """Compute statistics for each batch"""
        if data is None:
            raise ValueError("No data provided for computing statistics")
            
        # Group by Batch ID and compute statistics
        stats = data.groupby('Batch ID').agg({
            'DO (%)': ['mean', 'std', 'min', 'max'],
            'pH': ['mean', 'std', 'min', 'max'],
            'OD': ['mean', 'std', 'min', 'max'],
            'Temperature (deg C)': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.rename(columns={'Batch ID_': 'Batch ID'})
        
        return stats
        
    def prepare_data(self, scale, target):
        """Prepare data for modeling"""
        if self.process_data is None or self.initial_conditions is None or self.process_kpis is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
            
        try:
            # Subset data by scale
            subset_data = self.subset_by_scale(scale)
            print(f"\nSubsetting data for scale: {scale}")
            print("Shape before subsetting:", self.process_data.shape)
            print("Shape after subsetting:", subset_data.shape)
            
            # Compute batch statistics
            batch_stats = self.compute_batch_statistics(subset_data)
            print("\nBatch statistics shape:", batch_stats.shape)
            
            # Merge with initial conditions
            merged_data = pd.merge(batch_stats, self.initial_conditions, on='Batch ID', how='inner')
            print("\nShape after merging initial conditions:", merged_data.shape)
            
            # Add target
            if target == 'Final OD (OD 600)':
                target_col = 'Final OD'
            else:
                target_col = 'GFPuv (g/L)'
                
            final_data = pd.merge(merged_data, self.process_kpis[['Batch ID', target_col]], 
                                on='Batch ID', how='inner')
            print("\nShape after adding target:", final_data.shape)
            
            # Drop rows with missing values
            final_data = final_data.dropna()
            print("\nShape after dropping NA:", final_data.shape)
            
            # Prepare features and target
            feature_cols = [col for col in final_data.columns if col not in ['Batch ID', target_col]]
            X = final_data[feature_cols]
            y = final_data[target_col]
            
            # Normalize features
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            
            return X, y, final_data
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
            
    def display_data_summary(self, data, target):
        """Display summary of the data"""
        if data is None:
            raise ValueError("No data provided for summary")
            
        print("\nData Summary:")
        print(f"Number of samples: {len(data)}")
        print(f"Number of features: {len(data.columns) - 1}")  # Excluding target
        print("\nSummary statistics:")
        print(data.describe())
        print(f"\nTarget ({target}) distribution:")
        print(data[target].describe())
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Display data summary in Streamlit
        st.subheader('Data Summary')
        
        # Display data shape
        st.write(f"Number of samples: {len(data)}")
        st.write(f"Number of features: {len(data.columns) - 1}")  # Excluding target
        
        # Display summary statistics
        st.subheader('Summary Statistics')
        st.write(data.describe())
        
        # Display target distribution
        st.subheader(f'Target Distribution ({target})')
        fig, ax = plt.subplots()
        data[target].hist(ax=ax, bins=30)
        st.pyplot(fig)
        
        # Display missing values
        st.subheader('Missing Values')
        st.write(data.isnull().sum()) 