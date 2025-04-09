import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelDevelopment:
    def __init__(self, file_path):
        """Initialize the ModelDevelopment class"""
        self.file_path = file_path
        self.process_data = None
        self.initial_conditions = None
        self.process_kpis = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load data from Excel file"""
        try:
            # Get the absolute path to the project root directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            data_path = os.path.join(project_root, 'Wrangled_Combined_Batch_Dataset.xlsx')
            
            # Debug information
            print(f"Project root directory: {project_root}")
            print(f"Looking for data file at: {data_path}")
            print(f"File exists: {os.path.exists(data_path)}")
            
            if not os.path.exists(data_path):
                print(f"Data file not found at: {data_path}")
                # List files in the project root directory
                print("Files in project root directory:")
                for file in os.listdir(project_root):
                    if file.endswith('.xlsx'):
                        print(f"- {file}")
                return False
            
            # Load the Excel file
            self.process_data = pd.read_excel(data_path, sheet_name='Process Data')
            self.initial_conditions = pd.read_excel(data_path, sheet_name='Initial Conditions')
            self.process_kpis = pd.read_excel(data_path, sheet_name='Process KPIs')
            
            print("Data loaded successfully!")
            print(f"Process Data shape: {self.process_data.shape}")
            print(f"Initial Conditions shape: {self.initial_conditions.shape}")
            print(f"Process KPIs shape: {self.process_kpis.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def prepare_data(self, scale, target):
        """Prepare data for modeling"""
        try:
            # Subset data by scale
            subset_data = self.process_data[self.process_data['Scale'] == scale].copy()
            print(f"\nSubsetting data for scale: {scale}")
            print(f"Shape after subsetting: {subset_data.shape}")
            print("Columns after subsetting:", subset_data.columns.tolist())
            
            # Forward fill missing values within each batch
            print("\nBefore forward fill - checking Batch ID presence:", 'Batch ID' in subset_data.columns)
            # Store Batch ID separately
            batch_ids = subset_data['Batch ID'].copy()
            # Forward fill other columns
            subset_data = subset_data.groupby('Batch ID').ffill()
            # Restore Batch ID
            subset_data['Batch ID'] = batch_ids
            print("After forward fill - checking Batch ID presence:", 'Batch ID' in subset_data.columns)
            print("Applied forward fill within each batch")
            
            # Compute batch statistics
            numeric_cols = ['DO (%)', 'pH', 'OD', 'Temperature (deg C)']
            agg_dict = {}
            for col in numeric_cols:
                if col in subset_data.columns:
                    agg_dict[col.strip()] = ['mean', 'std', 'min', 'max']
            
            print("\nColumns to aggregate:", list(agg_dict.keys()))
            # Include Batch ID in aggregation to preserve it
            batch_stats = subset_data.groupby('Batch ID', as_index=False).agg({
                **agg_dict,
                'Batch ID': 'first'  # Keep the first value of Batch ID
            })
            
            # Flatten column names
            batch_stats.columns = ['Batch ID' if col == ('Batch ID', 'first') else '_'.join(col).strip() for col in batch_stats.columns]
            print("\nBatch stats columns after flattening:", batch_stats.columns.tolist())
            
            # Merge with initial conditions and forward fill any missing values
            merged_data = pd.merge(batch_stats, self.initial_conditions, on='Batch ID', how='inner')
            merged_data = merged_data.ffill()
            print(f"Shape after merging initial conditions: {merged_data.shape}")
            
            # Add target
            target_col = target.strip()  # Use the target name directly
            final_data = pd.merge(merged_data, 
                                self.process_kpis[['Batch ID', target_col]], 
                                on='Batch ID', how='inner')
            print(f"Shape after adding target: {final_data.shape}")
            
            # Forward fill any remaining missing values before final dropna
            final_data = final_data.ffill()
            
            # Drop any remaining rows with missing values (should be minimal now)
            rows_before = len(final_data)
            final_data = final_data.dropna()
            rows_after = len(final_data)
            if rows_before != rows_after:
                print(f"Dropped {rows_before - rows_after} rows with missing values that couldn't be forward filled")
            print(f"Shape after handling missing values: {final_data.shape}")
            
            # Identify numeric and categorical columns
            exclude_cols = ['Batch ID', target_col, 'Scale', 'Date']
            feature_cols = [col for col in final_data.columns if col not in exclude_cols]
            
            numeric_cols = final_data[feature_cols].select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = final_data[feature_cols].select_dtypes(include=['object']).columns
            
            print("\nNumeric features:", len(numeric_cols))
            print("Categorical features:", len(categorical_cols))
            
            # Prepare features
            X = final_data[feature_cols].copy()
            y = final_data[target_col]
            
            # Encode categorical variables
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            
            # Scale numeric features
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            
            return X, y, feature_cols
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
            
    def select_features(self, X, y, k=10):
        """Select top k features using f_regression"""
        try:
            # Select top k features
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names and scores
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = selector.scores_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_scores
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            print("\nSelected", len(selected_features), "features:")
            print(selected_features)
            
            # Return selected features and importance dataframe
            return X[selected_features], importance_df
            
        except Exception as e:
            print(f"Error selecting features: {str(e)}")
            raise
            
    def train_models(self, X, y, scale, target):
        """Train and evaluate models"""
        try:
            # Define models and their parameter grids
            models = {
                'RandomForest': (RandomForestRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None],
                    'min_samples_split': [2]
                }),
                'XGBoost': (xgb.XGBRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1, 0.3]
                }),
                'SVR': (SVR(), {
                    'kernel': ['linear', 'rbf'],
                    'C': [1, 10],
                    'epsilon': [0.1, 0.2]
                }),
                'PLS': (PLSRegression(n_components=5), {
                    'n_components': [2, 3, 4, 5],
                    'scale': [True]
                })
            }
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Train and evaluate each model
            for model_name, (model, param_grid) in models.items():
                print(f"\nTraining {model_name}...")
                
                # Perform grid search
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X, y)
                
                # Get best model and predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X)
                
                # Calculate metrics
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                r2 = grid_search.score(X, y)
                
                print(f"{model_name} Results:")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R2 Score: {r2:.4f}")
                
                # For PLS, print additional information
                if model_name == 'PLS':
                    n_components = best_model.n_components
                    # Calculate explained variance for PLS
                    X_transformed = best_model.transform(X)
                    X_reconstructed = best_model.inverse_transform(X_transformed)
                    total_var = np.var(X, axis=0).sum()
                    explained_var = 1 - np.var(X - X_reconstructed, axis=0).sum() / total_var
                    print(f"Number of components: {n_components}")
                    print(f"Explained variance: {explained_var * 100:.2f}%")
                
                # Save model
                scale_clean = scale.replace(' ', '_')
                target_clean = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
                model_path = f"models/{model_name}_{scale_clean}_{target_clean}.joblib"
                joblib.dump(best_model, model_path)
                print(f"Saved {model_name} model to {model_path}")
                
            # Save scaler and label encoders
            joblib.dump(self.scaler, f"models/scaler_{scale_clean}_{target_clean}.joblib")
            joblib.dump(self.label_encoders, f"models/label_encoders_{scale_clean}_{target_clean}.joblib")
            print(f"Saved scaler and label encoders")
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            raise

def main():
    """Main function to run the model development pipeline"""
    try:
        # Initialize pipeline
        pipeline = ModelDevelopment('Wrangled_Combined_Batch_Dataset.xlsx')
        
        # Load data
        if not pipeline.load_data():
            print("Failed to load data. Exiting...")
            return
            
        # Process each scale and target combination
        scales = ['1 mL', '30 L']
        targets = ['Final OD (OD 600)', 'GFPuv (g/L)']
        
        for scale in scales:
            print(f"\n{'-'*80}\n")
            for target in targets:
                print(f"\nProcessing {scale} - {target}")
                
                # Prepare data
                X, y, feature_cols = pipeline.prepare_data(scale, target)
                
                # Select features
                X_selected, _ = pipeline.select_features(X, y)
                
                # Train and evaluate models
                pipeline.train_models(X_selected, y, scale, target)
                
                print(f"Completed processing {scale} - {target}")
                print(f"\n{'-'*80}\n")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 