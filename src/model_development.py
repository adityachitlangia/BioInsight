import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
from sklearn.svm import SVR, OneClassSVM
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn import metrics

class ModelDevelopment:
    def __init__(self, file_path):
        """Initialize the ModelDevelopment class"""
        self.file_path = file_path
        self.process_data = None
        self.initial_conditions = None
        self.process_kpis = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.scalers = {}
        self.outlier_models = {}
        self.feature_selectors = {}
        
        # Create directories for model storage
        os.makedirs('models', exist_ok=True)
        
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
        """Prepare data for modeling for a specific scale and target"""
        # Subset data for the specified scale
        print(f"\nSubsetting data for scale: {scale}")
        subset_data = self.process_data[self.process_data['Scale'] == scale].copy()
        print(f"Shape after subsetting: {subset_data.shape}")
        print(f"Columns after subsetting: {subset_data.columns.tolist()}")
        
        # Check if Batch ID column exists before forward fill
        print(f"\nBefore forward fill - checking Batch ID presence: {'Batch ID' in subset_data.columns}")
        
        # Store Batch ID before operation that might remove it
        batch_ids = subset_data['Batch ID'].copy()
        
        # Forward fill missing values within each batch
        filled_data = subset_data.groupby('Batch ID').ffill()
        
        # Restore Batch ID column
        filled_data['Batch ID'] = batch_ids
        
        # Check if Batch ID column exists after forward fill
        print(f"After forward fill - checking Batch ID presence: {'Batch ID' in filled_data.columns}")
        print("Applied forward fill within each batch")
        
        # Compute statistics for numeric columns for each batch
        numeric_cols = ['DO (%)', 'pH', 'OD', 'Temperature (deg C)']
        print(f"\nColumns to aggregate: {numeric_cols}")
        
        # Function to compute statistics for a batch
        def compute_batch_stats(batch_df):
            stats = {}
            for col in numeric_cols:
                if col in batch_df.columns:
                    stats[f'{col}_mean'] = batch_df[col].mean()
                    stats[f'{col}_std'] = batch_df[col].std()
                    stats[f'{col}_min'] = batch_df[col].min()
                    stats[f'{col}_max'] = batch_df[col].max()
            return pd.Series(stats)
        
        # Calculate statistics for each batch
        batch_stats = filled_data.groupby('Batch ID').apply(compute_batch_stats).reset_index()
        print(f"\nBatch stats columns after flattening: {batch_stats.columns.tolist()}")
        
        # Merge with initial conditions
        ic_subset = self.initial_conditions[self.initial_conditions['Scale'] == scale].copy()
        merged_data = pd.merge(batch_stats, ic_subset, on='Batch ID', how='inner')
        print(f"Shape after merging initial conditions: {merged_data.shape}")
        
        # Add target from process KPIs
        kpi_subset = self.process_kpis[self.process_kpis['Scale'] == scale].copy()
        target_col = target
        merged_data = pd.merge(merged_data, kpi_subset[['Batch ID', target_col]], on='Batch ID', how='inner')
        print(f"Shape after adding target: {merged_data.shape}")
        
        # Handle missing values
        merged_data = merged_data.ffill()
        print(f"Shape after handling missing values: {merged_data.shape}")
        
        return merged_data
            
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
            
    def train_outlier_detection_models(self, X, scale, target):
        """Train outlier detection models and return them"""
        # Make a copy of X to avoid modifying the original
        X_copy = X.copy()
        
        # Format target name for file paths
        target_formatted = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        scale_formatted = scale.replace(' ', '_')
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_copy)
        
        outlier_models = {}
        
        # 1. PCA-X for outlier detection
        pca = PCA(n_components=2)  # Using 2 components for visualization
        pca.fit(X_scaled)
        
        # Save explained variance for reporting
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA-X Results:")
        print(f"Explained variance: {explained_variance:.2%}")
        print(f"Number of components: {pca.n_components_}")
        
        # Save PCA model
        model_path = f"models/PCA_X_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(pca, model_path)
        print(f"Saved PCA-X model to {model_path}")
        outlier_models['PCA-X'] = pca
        
        # 2. OPLS (using PLSRegression as proxy)
        # Note: OPLS is not directly available in scikit-learn, so we use PLS as an approximation
        opls = PLSRegression(n_components=2, scale=True)
        opls.fit(X_scaled, np.zeros(X_scaled.shape[0]))  # Using dummy Y for unsupervised
        
        # Save OPLS model
        model_path = f"models/OPLS_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(opls, model_path)
        print(f"Saved OPLS model to {model_path}")
        outlier_models['OPLS'] = opls
        
        # 3. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(X_scaled)
        
        # Save Isolation Forest model
        model_path = f"models/IsolationForest_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(iso_forest, model_path)
        print(f"Saved Isolation Forest model to {model_path}")
        outlier_models['Isolation Forest'] = iso_forest
        
        # 4. One-Class SVM
        one_class_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
        one_class_svm.fit(X_scaled)
        
        # Save One-Class SVM model
        model_path = f"models/OneClassSVM_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(one_class_svm, model_path)
        print(f"Saved One-Class SVM model to {model_path}")
        outlier_models['One-Class SVM'] = one_class_svm
        
        # Save the scaler for this scale/target combination for outlier detection
        scaler_path = f"models/outlier_scaler_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(scaler, scaler_path)
        
        return outlier_models

    def train_models(self, scale, target):
        """Train models for a specific scale and target"""
        print(f"\n{'-'*80}\n\nProcessing {scale} - {target}\n")
        
        # Prepare data
        data = self.prepare_data(scale, target)
        if data is None or len(data) == 0:
            print(f"No data available for {scale} - {target}")
            return False
        
        # Format target name for file paths
        target_formatted = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        scale_formatted = scale.replace(' ', '_')
        
        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Drop problematic columns: Date and any column that might have datetime data
        date_columns = [col for col in X.columns if 'Date' in col or X[col].dtype.name in ['datetime64[ns]', 'datetime64']]
        print(f"Dropping date columns: {date_columns}")
        if date_columns:
            X = X.drop(columns=date_columns)
        
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Prepare data for model training
        # Encode categorical features
        X_encoded = X.copy()
        label_encoders = {}
        
        for cat_col in categorical_features:
            le = LabelEncoder()
            X_encoded[cat_col] = le.fit_transform(X_encoded[cat_col].astype(str))
            label_encoders[cat_col] = le
        
        # Feature selection
        selector = SelectKBest(f_regression, k=10)
        X_selected = selector.fit_transform(X_encoded, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [X_encoded.columns[i] for i in selected_indices]
        
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features)
        
        # Create a DataFrame with only selected features
        X_final = X_encoded[selected_features]
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_final)
        
        # Save scaler for later use
        scaler_path = f"models/scaler_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Save feature selector and label encoders
        selector_path = f"models/selector_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(selector, selector_path)
        
        encoders_path = f"models/encoders_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(label_encoders, encoders_path)
        
        # Save selected feature names
        feature_names_path = f"models/features_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(selected_features, feature_names_path)
        
        # Train outlier detection models on the selected features
        self.outlier_models[(scale, target)] = self.train_outlier_detection_models(X_final, scale, target)
        
        # Train the models
        models = {}
        
        # 1. RandomForest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)
        best_rf = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = best_rf.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print("\nRandomForest Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save the model
        model_path = f"models/RandomForest_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(best_rf, model_path)
        print(f"Saved RandomForest model to {model_path}")
        models['RandomForest'] = best_rf
        
        # 2. XGBoost
        try:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5]
            }
            model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_scaled, y)
            best_xgb = grid_search.best_estimator_
            
            # Evaluate the model
            y_pred = best_xgb.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            print("\nXGBoost Results:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            
            # Save the model
            model_path = f"models/XGBoost_{scale_formatted}_{target_formatted}.joblib"
            joblib.dump(best_xgb, model_path)
            print(f"Saved XGBoost model to {model_path}")
            models['XGBoost'] = best_xgb
        except ImportError:
            print("XGBoost not available, skipping XGBoost training.")
        
        # 3. SVR
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [1, 10],
            'epsilon': [0.1, 0.2]
        }
        model = SVR()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)
        best_svr = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = best_svr.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print("\nSVR Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save the model
        model_path = f"models/SVR_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(best_svr, model_path)
        print(f"Saved SVR model to {model_path}")
        models['SVR'] = best_svr
        
        # 4. PLS Regression
        param_grid = {
            'n_components': [2, 3, 5, 10],
            'scale': [True, False]
        }
        model = PLSRegression()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)
        best_pls = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = best_pls.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred.ravel()))
        r2 = r2_score(y, y_pred.ravel())
        
        # Get number of components and explained variance
        n_components = best_pls.n_components
        explained_variance = np.sum(best_pls.explained_variance_ratio_) if hasattr(best_pls, 'explained_variance_ratio_') else None
        
        print("\nPLS Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Number of components: {n_components}")
        if explained_variance:
            print(f"Explained variance: {explained_variance:.2%}")
        
        # Save the model
        model_path = f"models/PLS_{scale_formatted}_{target_formatted}.joblib"
        joblib.dump(best_pls, model_path)
        print(f"Saved PLS model to {model_path}")
        models['PLS'] = best_pls
        
        print("Saved scaler and label encoders")
        print(f"Completed processing {scale} - {target}")
        
        # Store the models for this scale and target
        self.models[(scale, target)] = models
        self.scalers[(scale, target)] = scaler
        self.label_encoders[(scale, target)] = label_encoders
        self.feature_selectors[(scale, target)] = selector
        
        return True

    def run_pipeline(self):
        """Run the full data processing and model training pipeline"""
        if not self.load_data():
            print("Failed to load data. Exiting...")
            return False
        
        # Define scales and targets to model
        scales = ['1 mL', '30 L']
        targets = ['Final OD (OD 600)', 'GFPuv (g/L)']
        
        for scale in scales:
            for target in targets:
                self.train_models(scale, target)
                print(f"\n{'-'*80}\n")
        
        return True

def main():
    """Main function to run the pipeline"""
    try:
        # Create the pipeline with the dataset file path
        pipeline = ModelDevelopment('Wrangled_Combined_Batch_Dataset.xlsx')
        
        # Load data
        if not pipeline.load_data():
            print("Failed to load data. Exiting...")
            return
            
        # Define scales and targets to model
        scales = ['1 mL', '30 L']
        targets = ['Final OD (OD 600)', 'GFPuv (g/L)']
        
        # Run models for each scale and target
        for scale in scales:
            for target in targets:
                # Prepare data for modeling
                pipeline.train_models(scale, target)
                
                print(f"Completed processing {scale} - {target}")
                print(f"\n{'-'*80}\n")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 