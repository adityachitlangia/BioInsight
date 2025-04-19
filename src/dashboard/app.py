import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="BioInsight",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_development import ModelDevelopment

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stSelectbox, .stMultiselect {
        background-color: white;
        border-radius: 5px;
    }
    .stAlert {
        border-radius: 10px;
    }
    .header {
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    /* Remove card-like styling */
    .css-1d391kg {
        background-color: transparent;
        padding: 0;
        box-shadow: none;
    }
    .metric-card {
        background-color: transparent;
        padding: 0;
        box-shadow: none;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Load and return the data pipeline"""
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(project_root, 'Wrangled_Combined_Batch_Dataset.xlsx')
        
        # Debug information
        print(f"Project root directory: {project_root}")
        print(f"Looking for data file at: {data_path}")
        print(f"File exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            # List files in the project root directory
            print("Files in project root directory:")
            for file in os.listdir(project_root):
                if file.endswith('.xlsx'):
                    print(f"- {file}")
            return None
        
        # Create the model development pipeline
        pipeline = ModelDevelopment(data_path)
        if pipeline.load_data():
            return pipeline
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Show stack trace for debugging
        import traceback
        st.code(traceback.format_exc())
        return None

def plot_missing_values(df, title):
    """Plot missing values heatmap with interactive hover"""
    missing_data = df.isnull().astype(int)
    fig = px.imshow(missing_data,
                   title=title,
                   labels=dict(x="Features", y="Samples", color="Missing"),
                   color_continuous_scale='viridis')
    fig.update_layout(
        height=400,
        width=800,
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def plot_feature_importance(importance_df, title):
    """Plot feature importance with interactive hover"""
    fig = px.bar(importance_df.head(10),
                 x='importance',
                 y='feature',
                 title=title,
                 orientation='h',
                 labels={'importance': 'Importance Score', 'feature': 'Feature'},
                 color='importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(
        height=500,
        width=800,
        hovermode='y unified',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    )
    return fig

def plot_actual_vs_predicted(y_test, y_pred, model_name, target):
    """Plot actual vs predicted values with interactive hover"""
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        hovertemplate="<b>Actual:</b> %{x:.4f}<br><b>Predicted:</b> %{y:.4f}<extra></extra>"
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name}: Actual vs Predicted ({target})',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=500,
        width=800,
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def plot_batch_timeseries(data, feature, batch_ids=None):
    """Plot time series data for selected batches with interactive hover"""
    if batch_ids is None:
        batch_ids = data['Batch ID'].unique()[:5]
    
    fig = go.Figure()
    
    for batch_id in batch_ids:
        batch_data = data[data['Batch ID'] == batch_id]
        if not batch_data.empty and 'Culture Time (h)' in batch_data.columns and feature in batch_data.columns:
            fig.add_trace(go.Scatter(
                x=batch_data['Culture Time (h)'],
                y=batch_data[feature],
                name=f'Batch {batch_id}',
                mode='lines+markers',
                hovertemplate=f"<b>Time:</b> %{{x:.2f}} h<br><b>{feature}:</b> %{{y:.4f}}<br><b>Batch:</b> {batch_id}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f'{feature} vs Time by Batch',
        xaxis_title='Culture Time (h)',
        yaxis_title=feature,
        height=500,
        width=800,
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    return fig

def plot_pls_components(model, X, y, title):
    """Plot PLS components with interactive hover"""
    try:
        X_transformed = model.transform(X)
        
        # Create subplots
        fig = go.Figure()
        
        if model.n_components >= 2:
            # Plot first two components
            fig.add_trace(go.Scatter(
                x=X_transformed[:, 0],
                y=X_transformed[:, 1],
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Target Value')
                ),
                hovertemplate="<b>Component 1:</b> %{x:.4f}<br><b>Component 2:</b> %{y:.4f}<br><b>Target:</b> %{marker.color:.4f}<extra></extra>"
            ))
            fig.update_layout(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                title='First Two PLS Components'
            )
        else:
            # Plot single component against target
            fig.add_trace(go.Scatter(
                x=X_transformed[:, 0],
                y=y,
                mode='markers',
                hovertemplate="<b>Component 1:</b> %{x:.4f}<br><b>Target:</b> %{y:.4f}<extra></extra>"
            ))
            fig.update_layout(
                xaxis_title='Component 1',
                yaxis_title='Target Value',
                title='First PLS Component vs Target'
            )
        
        # Calculate explained variance
        var_explained = []
        total_var = np.var(X, axis=0).sum()
        X_transformed_full = model.transform(X)
        
        for i in range(model.n_components):
            X_transformed_i = np.zeros_like(X_transformed_full)
            X_transformed_i[:, :i+1] = X_transformed_full[:, :i+1]
            X_reconstructed_i = model.inverse_transform(X_transformed_i)
            unexplained_var = np.var(X - X_reconstructed_i, axis=0).sum()
            explained_var = (1 - unexplained_var / total_var) * 100
            var_explained.append(explained_var)
        
        # Create explained variance plot
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(
            x=list(range(1, model.n_components + 1)),
            y=var_explained,
            mode='lines+markers',
            hovertemplate="<b>Components:</b> %{x}<br><b>Explained Variance:</b> %{y:.2f}%<extra></extra>"
        ))
        
        fig_var.update_layout(
            title='Explained Variance by Components',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance (%)',
            height=400,
            width=800,
            hovermode='x unified',
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        return fig, fig_var
    except Exception as e:
        st.warning(f"Could not plot PLS components: {str(e)}")
        st.write(f"Debug info - X shape: {X.shape}, n_components: {model.n_components}")
        st.write(f"Model coefficients shape: {model.coef_.shape}")
        return None, None

def load_models_and_scalers(scale, target):
    """Load trained models, scalers, and other artifacts"""
    try:
        # Format target and scale for file paths
        target_formatted = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        scale_formatted = scale.replace(' ', '_')
        
        # Base path for models
        base_path = 'models'
        
        # Load regression models
        models = {}
        model_types = ['RandomForest', 'XGBoost', 'SVR', 'PLS']
        for model_type in model_types:
            model_path = os.path.join(base_path, f"{model_type}_{scale_formatted}_{target_formatted}.joblib")
            if os.path.exists(model_path):
                models[model_type] = joblib.load(model_path)
            else:
                st.warning(f"Model {model_type} not found at {model_path}")
        
        # Load outlier detection models
        outlier_models = {}
        outlier_model_types = ['PCA_X', 'OPLS', 'IsolationForest', 'OneClassSVM']
        for model_type in outlier_model_types:
            model_path = os.path.join(base_path, f"{model_type}_{scale_formatted}_{target_formatted}.joblib")
            if os.path.exists(model_path):
                outlier_models[model_type] = joblib.load(model_path)
            else:
                st.warning(f"Outlier model {model_type} not found at {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(base_path, f"scaler_{scale_formatted}_{target_formatted}.joblib")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Load outlier detection scaler
        outlier_scaler_path = os.path.join(base_path, f"outlier_scaler_{scale_formatted}_{target_formatted}.joblib")
        outlier_scaler = joblib.load(outlier_scaler_path) if os.path.exists(outlier_scaler_path) else None
        
        # Load feature names
        features_path = os.path.join(base_path, f"features_{scale_formatted}_{target_formatted}.joblib")
        features = joblib.load(features_path) if os.path.exists(features_path) else None
        
        # Load encoders
        encoders_path = os.path.join(base_path, f"encoders_{scale_formatted}_{target_formatted}.joblib")
        encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
        
        # Load feature selector
        selector_path = os.path.join(base_path, f"selector_{scale_formatted}_{target_formatted}.joblib")
        selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
        
        return {
            'models': models,
            'outlier_models': outlier_models,
            'scaler': scaler,
            'outlier_scaler': outlier_scaler,
            'features': features,
            'encoders': encoders,
            'selector': selector
        }
    except Exception as e:
        st.error(f"Error loading models and scalers: {str(e)}")
        return None

def plot_feature_distribution(data, feature):
    """Plot enhanced feature distribution with interactive elements and statistics"""
    # Create figure with subplots
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data[feature],
        nbinsx=20,
        name='Distribution',
        marker_color='#3498db',
        opacity=0.75,
        hovertemplate="<b>Value:</b> %{x:.4f}<br><b>Count:</b> %{y}<extra></extra>"
    ))
    
    # Add KDE line
    kde = ff.create_distplot([data[feature].dropna()], [feature], show_hist=False, show_rug=False)
    fig.add_trace(go.Scatter(
        x=kde.data[0].x,
        y=kde.data[0].y,
        mode='lines',
        name='Density',
        line=dict(color='#e74c3c', width=2),
        hovertemplate="<b>Value:</b> %{x:.4f}<br><b>Density:</b> %{y:.4f}<extra></extra>"
    ))
    
    # Calculate statistics
    stats = data[feature].describe()
    mean = stats['mean']
    std = stats['std']
    min_val = stats['min']
    max_val = stats['max']
    
    # Add mean line
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="green",
        annotation_text="Mean",
        annotation_position="top right"
    )
    
    # Add standard deviation lines
    fig.add_vline(
        x=mean - std,
        line_dash="dot",
        line_color="orange",
        annotation_text="-1σ",
        annotation_position="top right"
    )
    fig.add_vline(
        x=mean + std,
        line_dash="dot",
        line_color="orange",
        annotation_text="+1σ",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Distribution of {feature}",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis_title=feature,
        yaxis_title="Count",
        height=600,
        width=800,
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        )
    )
    
    # Create statistics table
    stats_table = pd.DataFrame({
        'Statistic': ['Mean', 'Standard Deviation', 'Minimum', 'Maximum', 'Count', 'Missing Values'],
        'Value': [
            f"{mean:.4f}",
            f"{std:.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}",
            f"{stats['count']}",
            f"{data[feature].isnull().sum()}"
        ]
    })
    
    return fig, stats_table

def plot_dimension_reduction(X, outlier_model, model_name, title):
    """Plot the PCA-X or OPLS results for outlier detection visualization"""
    try:
        if model_name in ['PCA_X', 'OPLS']:
            # Transform data using the model
            X_transformed = outlier_model.transform(X)
            
            # Create a DataFrame with the transformed data
            component_names = [f"Component 1", f"Component 2"]
            df_transformed = pd.DataFrame(X_transformed[:, :2], columns=component_names)
            
            # Add the original index as a column
            df_transformed['Sample'] = [f"Sample {i+1}" for i in range(len(df_transformed))]
            
            # Create a scatter plot using Plotly
            fig = px.scatter(
                df_transformed, 
                x=component_names[0], 
                y=component_names[1],
                hover_name='Sample',
                title=title,
                labels={component_names[0]: component_names[0], component_names[1]: component_names[1]},
                height=600
            )
            
            # Add a confidence ellipse if enough points
            if len(df_transformed) > 5:
                # Calculate center and standard deviation
                center_x = df_transformed[component_names[0]].mean()
                center_y = df_transformed[component_names[1]].mean()
                std_x = df_transformed[component_names[0]].std() * 2  # 95% confidence interval
                std_y = df_transformed[component_names[1]].std() * 2
                
                # Add confidence ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                x = center_x + std_x * np.cos(theta)
                y = center_y + std_y * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash'),
                    name='95% Confidence Interval'
                ))
            
            return fig
        else:
            st.warning(f"Model {model_name} is not supported for dimension reduction plots.")
            return None
    except Exception as e:
        st.error(f"Error plotting dimension reduction: {str(e)}")
        return None

def plot_outlier_scores(X, outlier_models, selected_models, title):
    """Plot outlier scores for selected models"""
    try:
        results = {}
        scores_df = pd.DataFrame()
        
        # Get outlier scores for each selected model
        for model_name in selected_models:
            if model_name in outlier_models:
                model = outlier_models[model_name]
                
                # Get scores based on model type
                if model_name == 'IsolationForest':
                    # For Isolation Forest, higher scores (closer to 1) are inliers
                    scores = model.decision_function(X)
                    scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to 0-1
                elif model_name == 'OneClassSVM':
                    # For One-Class SVM, higher scores (positive) are inliers
                    scores = model.decision_function(X)
                    scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to 0-1
                else:
                    continue
                
                # Store scores
                scores_df[model_name] = scores
                results[model_name] = scores
        
        if scores_df.empty:
            st.warning("No scores available for the selected models.")
            return None
        
        # Add sample index
        scores_df['Sample'] = [f"Sample {i+1}" for i in range(len(scores_df))]
        
        # Melt the DataFrame for Plotly
        melted_df = pd.melt(scores_df, id_vars=['Sample'], var_name='Model', value_name='Score')
        
        # Create a box plot using Plotly
        box_fig = px.box(
            melted_df, 
            x='Model', 
            y='Score',
            title=f"{title} - Outlier Scores Distribution",
            height=400
        )
        
        # Create a scatter plot for individual samples
        scatter_fig = px.scatter(
            melted_df, 
            x='Sample', 
            y='Score',
            color='Model',
            title=f"{title} - Outlier Scores by Sample",
            height=500
        )
        
        return {'box': box_fig, 'scatter': scatter_fig, 'scores': results}
    except Exception as e:
        st.error(f"Error plotting outlier scores: {str(e)}")
        return None

def main():
    # Custom header
    st.markdown("""
        <div style="background-color: #2c3e50; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0;">🧪 BioInsight</h1>
            <p style="color: #ecf0f1; text-align: center; margin: 0;">Data Analysis and Model Results Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = load_data()

    if st.session_state.pipeline is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return

    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("""
            <div style="background-color: #2c3e50; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color: white; text-align: center; margin: 0;">Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Go to",
            ["Data Processing", "Feature Selection", "Model Results", "Outlier Detection"],
            label_visibility="collapsed"
        )

    # Data Processing Section
    if page == "Data Processing":
        st.markdown('<h2 class="header">Data Processing Analysis</h2>', unsafe_allow_html=True)
        
        # Scale selection with improved styling
        scale = st.selectbox(
            "Select Scale",
            ["1 mL", "30 L"],
            key="scale_select"
        )
        
        # Show original data statistics
        st.markdown('<h3 class="header">Original Data Overview</h3>', unsafe_allow_html=True)
        subset_data = st.session_state.pipeline.process_data[
            st.session_state.pipeline.process_data['Scale'] == scale].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Shape:", subset_data.shape)
            st.write("Missing Values Before Forward Fill:")
            st.write(subset_data.isnull().sum())
            
        with col2:
            st.plotly_chart(plot_missing_values(subset_data, 
                f"Missing Values Heatmap - {scale} (Before Forward Fill)"), 
                use_container_width=True)
        
        # Show data after forward fill
        st.markdown('<h3 class="header">Data After Forward Fill</h3>', unsafe_allow_html=True)
        batch_ids = subset_data['Batch ID'].copy()
        filled_data = subset_data.groupby('Batch ID').ffill()
        filled_data['Batch ID'] = batch_ids
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("Missing Values After Forward Fill:")
            st.write(filled_data.isnull().sum())
            
        with col4:
            st.plotly_chart(plot_missing_values(filled_data, 
                f"Missing Values Heatmap - {scale} (After Forward Fill)"), 
                use_container_width=True)
        
        # Show batch statistics
        st.markdown('<h3 class="header">Batch Statistics</h3>', unsafe_allow_html=True)
        numeric_cols = ['DO (%)', 'pH', 'OD', 'Temperature (deg C)']
        stats_data = filled_data[numeric_cols].describe()
        st.write(stats_data)
        
        # Time series analysis
        st.markdown('<h3 class="header">Time Series Analysis</h3>', unsafe_allow_html=True)
        
        time_series_features = ['DO (%)', 'pH', 'OD', 'Temperature (deg C)', 'kLa (1/h)']
        available_features = [f for f in time_series_features if f in filled_data.columns]
        
        if available_features:
            feature = st.selectbox(
                "Select Feature for Time Series Plot",
                available_features,
                key="feature_select"
            )
            
            unique_batches = sorted(filled_data['Batch ID'].unique())
            selected_batches = st.multiselect(
                "Select Batches to Display",
                options=unique_batches,
                default=unique_batches[:3],
                key="batch_select"
            )
            
            if selected_batches:
                if 'Culture Time (h)' in filled_data.columns and feature in filled_data.columns:
                    st.plotly_chart(plot_batch_timeseries(filled_data, feature, selected_batches),
                        use_container_width=True)
                else:
                    st.error(f"Required columns 'Culture Time (h)' and '{feature}' not found in data")
        else:
            st.warning("No time series features available in the data")

    # Feature Selection Section
    elif page == "Feature Selection":
        st.header("Feature Selection Analysis")
        
        # Scale and target selection
        scale = st.selectbox("Select Scale", ["1 mL", "30 L"])
        target = st.selectbox("Select Target", 
            ["Final OD (OD 600)", "GFPuv (g/L)"])
        
        try:
            # Prepare data
            data = st.session_state.pipeline.prepare_data(scale, target)
            if data is None or len(data) == 0:
                st.error(f"No data available for {scale} with target {target}")
                return
                
            # Format X and y for feature selection
            X = data.drop(columns=[target])
            y = data[target]
            
            # Drop date columns
            date_columns = [col for col in X.columns if 'Date' in col or X[col].dtype.name in ['datetime64[ns]', 'datetime64']]
            if date_columns:
                X = X.drop(columns=date_columns)
            
            # Make a copy for encoding
            X_encoded = X.copy()
            
            # Identify columns that need encoding (not just object dtype, but anything non-numeric)
            numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = [col for col in X_encoded.columns if col not in numeric_cols]
            
            st.write(f"Numerical columns: {len(numeric_cols)}")
            st.write(f"Categorical columns to encode: {len(categorical_cols)}, {categorical_cols}")
            
            # Encode all non-numeric columns
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values if present
                if X_encoded[col].isna().any():
                    X_encoded[col] = X_encoded[col].fillna('missing')
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            
            # Verify all columns are now numeric
            if not all(X_encoded.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                non_numeric = [col for col in X_encoded.columns if not np.issubdtype(X_encoded[col].dtype, np.number)]
                st.error(f"Some columns are still non-numeric after encoding: {non_numeric}")
                return
            
            # Now apply feature selection
            selector = SelectKBest(f_regression, k=10)
            X_selected_array = selector.fit_transform(X_encoded, y)
            
            # Get selected feature names and scores
            selected_features = X_encoded.columns[selector.get_support()].tolist()
            feature_scores = selector.scores_
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': X_encoded.columns,
                'importance': feature_scores
            }).sort_values('importance', ascending=False)
            
            # Convert X_selected back to DataFrame with feature names
            X_selected_df = pd.DataFrame(X_selected_array, columns=selected_features)
            
            # Show feature importance
            st.subheader("Feature Importance")
            st.plotly_chart(plot_feature_importance(feature_importance, 
                f"Top 10 Important Features - {scale}, {target}"),
                use_container_width=True)
            
            # Show correlation matrix
            st.subheader("Feature Correlation Matrix")
            corr_matrix = X_selected_df.corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=np.around(corr_matrix.values, decimals=2),
                colorscale='RdBu',
                showscale=True
            )
            fig.update_layout(
                title=f"Correlation Matrix - Selected Features ({scale}, {target})",
                height=800,
                width=800
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature distributions using enhanced plotting
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select Feature", selected_features)
            
            # Create two columns for plot and statistics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, stats_table = plot_feature_distribution(X_selected_df, feature)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                    <div style="background-color: white; padding: 1rem; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 2rem;">
                        <h3 style="color: #2c3e50; margin-top: 0;">Statistics</h3>
                """, unsafe_allow_html=True)
                st.table(stats_table)
                st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in feature selection: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Model Results Section
    elif page == "Model Results":
        st.header("Model Results and Comparison")
        
        # Scale and target selection
        scale = st.selectbox("Select Scale", ["1 mL", "30 L"])
        target = st.selectbox("Select Target", 
            ["Final OD (OD 600)", "GFPuv (g/L)"])
        
        try:
            # Prepare data
            data = st.session_state.pipeline.prepare_data(scale, target)
            if data is None or len(data) == 0:
                st.error(f"No data available for {scale} with target {target}")
                return
                
            # Format X and y for models
            X = data.drop(columns=[target])
            y = data[target]
            
            # Drop date columns and other problematic columns
            date_columns = [col for col in X.columns if 'Date' in col or X[col].dtype.name in ['datetime64[ns]', 'datetime64']]
            if date_columns:
                X = X.drop(columns=date_columns)
            
            # Load models and scalers
            models_and_scalers = load_models_and_scalers(scale, target)
            
            if models_and_scalers is None:
                st.error("Failed to load models. Please ensure models have been trained.")
                return
            
            # Get features used for modeling
            features = models_and_scalers['features']
            if features is None:
                st.error("Feature list not found for the selected model.")
                return
                
            # Select only the features used by the models
            if all(feat in X.columns for feat in features):
                X_selected = X[features]
            else:
                st.warning("Some model features are not present in the data. Using all available features.")
                X_selected = X
                
            # Encode categorical features if needed
            encoders = models_and_scalers['encoders']
            if encoders:
                for col in X_selected.columns:
                    if col in encoders:
                        X_selected[col] = encoders[col].transform(X_selected[col].astype(str))
            
            # Scale the features if needed
            scaler = models_and_scalers['scaler']
            if scaler:
                X_scaled = scaler.transform(X_selected)
            else:
                X_scaled = X_selected.values
            
            # Evaluate each model
            for model_name, model in models_and_scalers['models'].items():
                try:
                    # Make predictions
                    if model_name == 'PLS':
                        # For PLS, use the scaled data
                        y_pred = model.predict(X_scaled)
                        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                            y_pred = y_pred.ravel()
                    else:
                        y_pred = model.predict(X_scaled)
                    
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    r2 = r2_score(y, y_pred)
                    
                    # Display results
                    st.subheader(f"{model_name} Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Model Performance:")
                        st.write(f"RMSE: {rmse:.4f}")
                        st.write(f"R2 Score: {r2:.4f}")
                        
                        # Show feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            importance = pd.DataFrame({
                                'feature': X_selected.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            st.write("\nTop 5 Important Features:")
                            st.write(importance.head())
                        
                        # Show PLS-specific information
                        if model_name == 'PLS':
                            st.write("\nPLS Model Information:")
                            st.write(f"Number of components: {model.n_components}")
                            
                            # Calculate explained variance
                            X_transformed = model.transform(X_scaled)
                            X_reconstructed = model.inverse_transform(X_transformed)
                            total_var = np.var(X_scaled, axis=0).sum()
                            explained_var = 1 - np.var(X_scaled - X_reconstructed, axis=0).sum() / total_var
                            st.write(f"Total explained variance: {explained_var * 100:.2f}%")
                            
                            # Show loadings
                            loadings = pd.DataFrame(
                                model.x_loadings_,
                                columns=[f'Component {i+1}' for i in range(model.n_components)],
                                index=X_selected.columns
                            )
                            st.write("\nComponent Loadings (top 5 features):")
                            st.write(loadings.abs().sum(axis=1).sort_values(ascending=False).head())
                    
                    with col2:
                        st.plotly_chart(plot_actual_vs_predicted(y, y_pred, model_name, target),
                            use_container_width=True)
                    
                    # Show PLS components plot
                    if model_name == 'PLS':
                        fig_components, fig_variance = plot_pls_components(model, X_scaled, y, 
                            f"PLS Components Analysis - {scale}, {target}")
                        if fig_components is not None:
                            st.plotly_chart(fig_components, use_container_width=True)
                            st.plotly_chart(fig_variance, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not evaluate {model_name} model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Interactive prediction
            if models_and_scalers['models']:
                st.subheader("Interactive Prediction")
                
                st.write("Enter raw measurements and we'll predict the target values:")
                
                # Get the original process data features
                raw_features = ['DO (%)', 'pH', 'OD', 'Temperature (deg C)']
                
                # Create input fields for raw measurements
                raw_inputs = {}
                col1, col2 = st.columns(2)
                
                with col1:
                    for feature in raw_features:
                        # Get statistics from the original data for reference
                        feature_data = st.session_state.pipeline.process_data[feature].dropna()
                        if not feature_data.empty:
                            mean_val = float(feature_data.mean())
                            min_val = float(feature_data.min())
                            max_val = float(feature_data.max())
                            raw_inputs[feature] = st.number_input(
                                f"{feature} Value",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                format="%.2f"
                            )
                
                with col2:
                    # Add initial conditions inputs
                    inoculation_od = st.number_input(
                        "Inoculation OD",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.1,
                        format="%.2f"
                    )
                    
                    initial_glucose = st.number_input(
                        "Initial Glucose (g/L)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        format="%.2f"
                    )
                    
                    iptg = st.number_input(
                        "IPTG (mM)",
                        min_value=0.0,
                        max_value=5.0,
                        value=1.0,
                        format="%.2f"
                    )
                
                if st.button("Predict"):
                    # Calculate the aggregated features
                    calculated_features = {}
                    
                    # Basic calculation of means, mins, maxs (assuming we have just one value per feature)
                    for feature in raw_features:
                        calculated_features[f"{feature}_mean"] = raw_inputs[feature]
                        calculated_features[f"{feature}_min"] = raw_inputs[feature]
                        calculated_features[f"{feature}_max"] = raw_inputs[feature]
                        # Standard deviation would be 0 for a single value
                        calculated_features[f"{feature}_std"] = 0.0
                    
                    # Add initial conditions
                    calculated_features["Inoculation OD"] = inoculation_od
                    calculated_features["Initial Glucose (g/L)"] = initial_glucose
                    calculated_features["IPTG (mM)"] = iptg
                    
                    # Add other required features with default values
                    for feature in X_selected.columns:
                        if feature not in calculated_features and feature != "Batch ID":
                            calculated_features[feature] = 0.0
                    
                    # Display the calculated features
                    with st.expander("View calculated aggregated features"):
                        st.write("These are the aggregated features calculated from your raw inputs:")
                        calc_features_df = pd.DataFrame([calculated_features])
                        st.dataframe(calc_features_df[X_selected.columns])
                    
                    # Select only the features needed by the model
                    input_df = calc_features_df[X_selected.columns]
                    
                    # Scale input features if needed
                    if scaler:
                        input_scaled = scaler.transform(input_df)
                    else:
                        input_scaled = input_df.values
                    
                    # Store predictions from all models
                    predictions = {}
                    prediction_values = []
                    
                    # Create columns for predictions
                    st.write("### Model Predictions")
                    model_cols = st.columns(len(models_and_scalers['models']))
                    
                    # Make predictions with each model
                    for i, (model_name, model) in enumerate(models_and_scalers['models'].items()):
                        try:
                            if model_name == 'PLS':
                                prediction = model.predict(input_scaled)[0]
                                if isinstance(prediction, np.ndarray):
                                    prediction = prediction[0]
                            else:
                                prediction = model.predict(input_scaled)[0]
                                
                            predictions[model_name] = prediction
                            prediction_values.append(prediction)
                            
                            # Display prediction in respective column with styling
                            with model_cols[i]:
                                st.metric(
                                    label=model_name,
                                    value=f"{prediction:.4f}"
                                )
                        except Exception as e:
                            with model_cols[i]:
                                st.error(f"Error: {str(e)}")
                    
                    # Add prediction validation section
                    st.write("### Prediction Validation")
                    
                    # 1. Calculate average and variance across models
                    if len(prediction_values) > 0:
                        mean_pred = np.mean(prediction_values)
                        std_pred = np.std(prediction_values)
                        
                        st.write(f"**Average prediction**: {mean_pred:.4f}")
                        st.write(f"**Standard deviation across models**: {std_pred:.4f}")
                        
                        # Calculate coefficient of variation to assess prediction consistency
                        if mean_pred != 0:
                            cv = (std_pred / mean_pred) * 100
                            st.write(f"**Coefficient of variation**: {cv:.2f}% {'(High variability across models)' if cv > 10 else '(Good consistency across models)'}")
                    
                    # 2. Find similar samples in the training data
                    st.write("#### Similarity Analysis")
                    st.write("Finding similar samples in the dataset for comparison:")
                    
                    # Calculate distances to each sample in the dataset
                    distances = []
                    for i in range(len(X_scaled)):
                        # Use Euclidean distance
                        dist = np.sqrt(np.sum((X_scaled[i] - input_scaled[0])**2))
                        distances.append(dist)
                    
                    # Create a DataFrame with distances
                    similarity_df = pd.DataFrame({
                        'Sample Index': range(len(X_scaled)),
                        'Distance': distances,
                        'Actual Target': y.values
                    })
                    
                    # Sort by distance and get top 5 most similar samples
                    similar_samples = similarity_df.sort_values('Distance').head(5)
                    
                    # Show the similar samples
                    st.write("Most similar samples in the dataset:")
                    similar_samples_with_features = pd.concat([
                        similar_samples, 
                        X.iloc[similar_samples['Sample Index']].reset_index(drop=True)
                    ], axis=1)
                    
                    st.dataframe(similar_samples_with_features)
                    
                    # Calculate mean of actual values from similar samples
                    mean_similar = similar_samples['Actual Target'].mean()
                    st.write(f"**Average target value of similar samples**: {mean_similar:.4f}")
                    
                    # Compare prediction with similar samples average
                    pred_diff = mean_pred - mean_similar
                    st.write(f"**Difference from similar samples average**: {pred_diff:.4f} ({'higher' if pred_diff > 0 else 'lower'})")
                    
                    # Visualization comparing predictions with similar samples
                    fig = go.Figure()
                    
                    # Add model predictions
                    for model_name, pred in predictions.items():
                        fig.add_trace(go.Scatter(
                            x=[model_name],
                            y=[pred],
                            mode='markers',
                            marker=dict(size=12, color='blue'),
                            name='Model predictions'
                        ))
                    
                    # Add actual values of similar samples
                    fig.add_trace(go.Scatter(
                        x=['Similar1', 'Similar2', 'Similar3', 'Similar4', 'Similar5'],
                        y=similar_samples['Actual Target'].values,
                        mode='markers',
                        marker=dict(size=10, color='green'),
                        name='Similar samples (actual)'
                    ))
                    
                    # Add mean line of similar samples
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=mean_similar,
                        x1=len(predictions) + 4.5,
                        y1=mean_similar,
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    # Add mean line of predictions
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=mean_pred,
                        x1=len(predictions) - 0.5,
                        y1=mean_pred,
                        line=dict(
                            color="blue",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    fig.update_layout(
                        title="Comparison of Model Predictions vs. Similar Samples",
                        xaxis_title="Model / Similar Sample",
                        yaxis_title=f"Predicted {target}",
                        legend_title="Data Source",
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if abs(pred_diff) < 0.1 * mean_similar:
                        st.success("✅ Prediction is consistent with similar samples in the dataset.")
                    elif abs(pred_diff) < 0.25 * mean_similar:
                        st.warning("⚠️ Prediction differs somewhat from similar samples, but may still be reasonable.")
                    else:
                        st.error("⚠️ Prediction differs significantly from similar samples. Consider reviewing input values.")
            
        except Exception as e:
            st.error(f"Error in model evaluation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.warning("Please ensure models have been trained and saved correctly.")

    # Outlier Detection Section
    elif page == "Outlier Detection":
        st.markdown('<h2 class="header">Outlier Detection Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            scale = st.selectbox(
                "Select Scale",
                ["1 mL", "30 L"],
                key="od_scale_select"
            )
        
        with col2:
            target = st.selectbox(
                "Select Target Variable",
                ["Final OD (OD 600)", "GFPuv (g/L)"],
                key="od_target_select"
            )
        
        # Load models for the selected scale and target
        model_data = load_models_and_scalers(scale, target)
        
        if model_data is None:
            st.error("Failed to load models. Please ensure models have been trained.")
            return
        
        # Prepare data for the selected scale
        st.markdown('<h3 class="header">Batch Data and Outlier Detection</h3>', unsafe_allow_html=True)
        st.info("This section allows you to identify potential outlier batches using different detection methods.")
        
        # Prepare data
        data = st.session_state.pipeline.prepare_data(scale, target)
        if data.empty:
            st.error(f"No data available for {scale} with target {target}")
            return
        
        # Display data summary
        st.write(f"Number of batches for analysis: {len(data)}")
        
        # Extract features used for modeling
        features = model_data['features']
        if features is None:
            st.error("Feature list not found for the selected model.")
            return
        
        X = data[features]
        
        # Encode categorical features if needed
        encoders = model_data['encoders']
        if encoders:
            X_encoded = X.copy()
            for col in X.columns:
                if col in encoders:
                    X_encoded[col] = encoders[col].transform(X_encoded[col].astype(str))
            X = X_encoded
        
        # Scale the features
        scaler = model_data['scaler']
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Get outlier detection models
        outlier_models = model_data['outlier_models']
        if not outlier_models:
            st.error("No outlier detection models found.")
            return
        
        # Model selection for visualization
        st.markdown('<h3 class="header">Outlier Detection Model Selection</h3>', unsafe_allow_html=True)
        
        # Select models to visualize
        available_models = list(outlier_models.keys())
        selected_models = st.multiselect(
            "Select models to compare",
            available_models,
            default=available_models[:1] if available_models else [],
            key="outlier_model_select"
        )
        
        if not selected_models:
            st.warning("Please select at least one model to visualize.")
            return
        
        # Visualization tabs
        st.markdown('<h3 class="header">Outlier Detection Visualizations</h3>', unsafe_allow_html=True)
        
        tabs = st.tabs(["Dimension Reduction", "Outlier Scores", "Comparative Analysis"])
        
        with tabs[0]:
            st.write("### Dimension Reduction Visualization")
            st.info("This visualization shows how samples are distributed in a reduced dimensional space.")
            
            # For each PCA or OPLS model selected
            for model_name in selected_models:
                if model_name in ['PCA_X', 'OPLS']:
                    model = outlier_models[model_name]
                    fig = plot_dimension_reduction(
                        X_scaled, 
                        model, 
                        model_name, 
                        f"{model_name} - {scale} - {target}"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.write("### Outlier Scores Visualization")
            st.info("This visualization shows outlier scores for each sample. Lower scores indicate potential outliers.")
            
            # Plot outlier scores for selected models
            outlier_scores = plot_outlier_scores(
                X_scaled,
                outlier_models,
                [m for m in selected_models if m in ['IsolationForest', 'OneClassSVM']],
                f"Outlier Analysis - {scale} - {target}"
            )
            
            if outlier_scores:
                st.plotly_chart(outlier_scores['box'], use_container_width=True)
                st.plotly_chart(outlier_scores['scatter'], use_container_width=True)
                
                # Identify potential outliers
                if 'scores' in outlier_scores:
                    st.write("### Potential Outliers")
                    st.info("Samples with outlier scores below threshold may be considered outliers.")
                    
                    # Threshold selection
                    threshold = st.slider("Outlier threshold", 0.0, 1.0, 0.1, 0.05)
                    
                    # Display potential outliers for each model
                    for model_name, scores in outlier_scores['scores'].items():
                        outliers = np.where(scores < threshold)[0]
                        if len(outliers) > 0:
                            st.write(f"#### {model_name} detected {len(outliers)} potential outliers:")
                            outlier_data = data.iloc[outliers].copy()
                            outlier_data['Outlier Score'] = scores[outliers]
                            st.dataframe(outlier_data)
                        else:
                            st.write(f"#### {model_name}: No outliers detected at threshold {threshold}")
        
        with tabs[2]:
            st.write("### Comparative Analysis")
            st.info("This visualization compares the outlier detection results across different models.")
            
            # Compare outlier detection across models
            comparison_models = [m for m in selected_models if m in ['IsolationForest', 'OneClassSVM']]
            
            if len(comparison_models) > 1 and 'scores' in outlier_scores:
                # Get scores for comparison
                scores_dict = outlier_scores['scores']
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame({
                    model: scores for model, scores in scores_dict.items()
                })
                
                # Add batch ID for reference
                comparison_df['Batch ID'] = data['Batch ID'].values
                
                # Plot correlation heatmap between model scores
                corr_matrix = comparison_df.drop(columns=['Batch ID']).corr()
                
                # Create heatmap with Plotly
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title=f"Correlation between model scores - {scale} - {target}",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model agreement analysis
                st.write("#### Model Agreement Analysis")
                
                # Set threshold for outlier detection
                agreement_threshold = st.slider(
                    "Agreement threshold", 
                    0.0, 1.0, 0.1, 0.05, 
                    key="agreement_threshold"
                )
                
                # Count how many models flag each sample as an outlier
                outlier_counts = pd.DataFrame()
                outlier_counts['Batch ID'] = data['Batch ID']
                
                for model in comparison_models:
                    outlier_counts[f"{model}_outlier"] = (scores_dict[model] < agreement_threshold).astype(int)
                
                # Add a column with the total number of models that flag each sample
                outlier_counts['Total Flags'] = outlier_counts[[f"{model}_outlier" for model in comparison_models]].sum(axis=1)
                
                # Create histogram of agreement counts
                agreement_fig = px.histogram(
                    outlier_counts, 
                    x='Total Flags',
                    title=f"Number of models flagging each sample as outlier (threshold: {agreement_threshold})",
                    labels={'Total Flags': 'Number of models flagging as outlier'},
                    height=400
                )
                st.plotly_chart(agreement_fig, use_container_width=True)
                
                # Display samples flagged by multiple models
                multi_flagged = outlier_counts[outlier_counts['Total Flags'] > 1]
                if not multi_flagged.empty:
                    st.write(f"#### {len(multi_flagged)} samples flagged by multiple models:")
                    
                    # Merge with original data for more context
                    multi_flagged_data = pd.merge(multi_flagged, data, on='Batch ID', how='inner')
                    st.dataframe(multi_flagged_data)
                else:
                    st.write("#### No samples flagged by multiple models")
            else:
                st.warning("Please select at least two outlier detection models for comparison.")

if __name__ == "__main__":
    main() 