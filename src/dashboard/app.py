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
        st.write(f"Project root directory: {project_root}")
        st.write(f"Looking for data file at: {data_path}")
        st.write(f"File exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            # List files in the project root directory
            st.write("Files in project root directory:")
            for file in os.listdir(project_root):
                if file.endswith('.xlsx'):
                    st.write(f"- {file}")
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
    """Load models, scaler, and label encoders for a given scale and target"""
    try:
        scale_clean = scale.replace(' ', '_')
        target_clean = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Load scaler and encoders first
        scaler = joblib.load(os.path.join(base_path, 'models', f"scaler_{scale_clean}_{target_clean}.joblib"))
        label_encoders = joblib.load(os.path.join(base_path, 'models', f"label_encoders_{scale_clean}_{target_clean}.joblib"))
        
        # Load models
        models = {}
        model_names = ['RandomForest', 'XGBoost', 'SVR', 'PLS']
        for model_name in model_names:
            model_path = os.path.join(base_path, 'models', f"{model_name}_{scale_clean}_{target_clean}.joblib")
            models[model_name] = joblib.load(model_path)
            
        return models, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading models and scalers: {str(e)}")
        return None, None, None

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
            ["Data Processing", "Feature Selection", "Model Results"],
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
            # Prepare data and select features
            X, y, feature_cols = st.session_state.pipeline.prepare_data(scale, target)
            X_selected, feature_importance = st.session_state.pipeline.select_features(X, y)
            
            # Show feature importance
            st.subheader("Feature Importance")
            st.plotly_chart(plot_feature_importance(feature_importance, 
                f"Top 10 Important Features - {scale}, {target}"),
                use_container_width=True)
            
            # Show correlation matrix
            st.subheader("Feature Correlation Matrix")
            corr_matrix = X_selected.corr()
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
            feature = st.selectbox("Select Feature", X_selected.columns)
            
            # Create two columns for plot and statistics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, stats_table = plot_feature_distribution(X_selected, feature)
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

    # Model Results Section
    else:
        st.header("Model Results and Comparison")
        
        # Scale and target selection
        scale = st.selectbox("Select Scale", ["1 mL", "30 L"])
        target = st.selectbox("Select Target", 
            ["Final OD (OD 600)", "GFPuv (g/L)"])
        
        try:
            # Prepare data
            X, y, feature_cols = st.session_state.pipeline.prepare_data(scale, target)
            X_selected, _ = st.session_state.pipeline.select_features(X, y)
            
            # Load models and scalers
            models, scaler, label_encoders = load_models_and_scalers(scale, target)
            
            if models is None:
                st.error("Failed to load models. Please ensure models have been trained.")
                return
            
            # Evaluate each model
            for model_name, model in models.items():
                try:
                    # Make predictions
                    if model_name == 'PLS':
                        # For PLS, we use all features since it handles feature selection internally
                        y_pred = model.predict(X_selected)
                    else:
                        y_pred = model.predict(X_selected)
                    
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
                            X_transformed = model.transform(X_selected)
                            X_reconstructed = model.inverse_transform(X_transformed)
                            total_var = np.var(X_selected, axis=0).sum()
                            explained_var = 1 - np.var(X_selected - X_reconstructed, axis=0).sum() / total_var
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
                        fig_components, fig_variance = plot_pls_components(model, X_selected, y, 
                            f"PLS Components Analysis - {scale}, {target}")
                        if fig_components is not None:
                            st.plotly_chart(fig_components, use_container_width=True)
                            st.plotly_chart(fig_variance, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not evaluate {model_name} model: {str(e)}")
            
            # Interactive prediction
            if models:
                st.subheader("Interactive Prediction")
                st.write("Select feature values for prediction:")
                
                input_features = {}
                for feature in X_selected.columns:
                    mean_val = float(X_selected[feature].mean())
                    std_val = float(X_selected[feature].std())
                    input_features[feature] = st.slider(
                        feature,
                        min_value=mean_val - 2*std_val,
                        max_value=mean_val + 2*std_val,
                        value=mean_val,
                        format="%.2f"
                    )
                
                if st.button("Predict"):
                    input_df = pd.DataFrame([input_features])
                    st.write("\nPredictions:")
                    for model_name, model in models.items():
                        try:
                            # All models can use the same input features since PLS handles feature selection internally
                            prediction = model.predict(input_df)[0]
                            st.write(f"{model_name}: {prediction:.4f}")
                        except Exception as e:
                            st.warning(f"Could not make prediction with {model_name}: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in model evaluation: {str(e)}")
            st.write("Please ensure models have been trained and saved correctly.")

if __name__ == "__main__":
    main() 