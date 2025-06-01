import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Advanced Regression Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .model-comparison {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB Connection
@st.cache_resource
def init_mongo_connection():
    """Initialize MongoDB connection"""
    try:
        # Try to get connection string from Streamlit secrets
        if "mongo" in st.secrets:
            connection_string = st.secrets["mongo"]["connection_string"]
            database_name = st.secrets["mongo"].get("database_name", "regression_analysis")
        else:
            # Fallback - you can replace this with your connection string
            connection_string = "mongodb+srv://tanmoy:tanmoy@cluster0.jnqr9nr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            database_name = "regression_analysis"
        
        client = pymongo.MongoClient(connection_string)
        # Test the connection
        client.admin.command('ismaster')
        db = client[database_name]
        return db
    except Exception as e:
        st.warning(f"MongoDB connection failed: {e}")
        st.info("Running in offline mode. Results will not be saved.")
        return None

# Initialize MongoDB
db = init_mongo_connection()

# Helper Functions
def calculate_adjusted_r2(r2, n_samples, n_features):
    """Calculate adjusted R²"""
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

def evaluate_model(y_true, y_pred, n_features):
    """Calculate all evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = calculate_adjusted_r2(r2, len(y_true), n_features)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Adj R²': adj_r2
    }

def save_to_mongodb(model_type, dataset_name, metrics, parameters):
    """Save model results to MongoDB"""
    if db is not None:
        try:
            collection = db["model_results"]
            document = {
                "timestamp": datetime.now(),
                "model_type": model_type,
                "dataset": dataset_name,
                "metrics": metrics,
                "parameters": parameters
            }
            collection.insert_one(document)
            return True
        except Exception as e:
            st.error(f"Failed to save to MongoDB: {e}")
            return False
    else:
        st.warning("MongoDB not available. Cannot save results.")
        return False

def load_from_mongodb():
    """Load previous results from MongoDB"""
    if db is not None:
        try:
            collection = db["model_results"]
            results = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
            return results
        except Exception as e:
            st.error(f"Failed to load from MongoDB: {e}")
            return []
    else:
        st.warning("MongoDB not available. Cannot load previous results.")
        return []

# Main App
def main():
    st.markdown('<h1 class="main-header">🔬 Advanced Regression Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Diabetes Dataset", "Ice Cream Sales", "Upload Custom Data"]
    )
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model Type",
        ["Ridge & Lasso Comparison", "Polynomial Regression", "All Models"]
    )
    
    # Load data based on selection
    if dataset_choice == "Diabetes Dataset":
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        X = df.drop('target', axis=1)
        y = df['target']
        st.sidebar.success("✅ Diabetes dataset loaded!")
        
    elif dataset_choice == "Ice Cream Sales":
        # Create sample ice cream data
        np.random.seed(42)
        temperature = np.random.uniform(-5, 40, 100)
        sales = 50 + 2.5 * temperature + 0.1 * temperature**2 + np.random.normal(0, 10, 100)
        df = pd.DataFrame({
            'Temperature (°C)': temperature,
            'Ice Cream Sales (units)': sales
        })
        X = df[['Temperature (°C)']]
        y = df['Ice Cream Sales (units)']
        st.sidebar.success("✅ Ice cream sales dataset loaded!")
        
    elif dataset_choice == "Upload Custom Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Custom dataset loaded!")
            
            # Let user select target variable
            target_col = st.sidebar.selectbox("Select target variable", df.columns)
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        else:
            st.warning("Please upload a CSV file to proceed.")
            return
    
    # Display dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Overview")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Features:** {len(X.columns)}")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📈 Data Visualization")
        if len(X.columns) == 1:
            fig = px.scatter(df, x=X.columns[0], y=y.name if hasattr(y, 'name') else 'target',
                           title="Feature vs Target")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Correlation heatmap for multiple features
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    # Split data
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training and Evaluation
    st.header("🤖 Model Training & Evaluation")
    
    if model_choice in ["Ridge & Lasso Comparison", "All Models"]:
        st.subheader("Ridge & Lasso Regression")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ridge_alpha = st.slider("Ridge Alpha", 0.01, 10.0, 1.0, 0.01)
        with col2:
            lasso_alpha = st.slider("Lasso Alpha", 0.01, 10.0, 1.0, 0.01)
        
        # Train models
        ridge_model = Ridge(alpha=ridge_alpha)
        lasso_model = Lasso(alpha=lasso_alpha)
        
        ridge_model.fit(X_train_scaled, y_train)
        lasso_model.fit(X_train_scaled, y_train)
        
        # Predictions
        ridge_pred = ridge_model.predict(X_test_scaled)
        lasso_pred = lasso_model.predict(X_test_scaled)
        
        # Evaluate models
        ridge_metrics = evaluate_model(y_test, ridge_pred, X_test.shape[1])
        lasso_metrics = evaluate_model(y_test, lasso_pred, X_test.shape[1])
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.write("### 🟦 Ridge Regression Results")
            for metric, value in ridge_metrics.items():
                st.metric(metric, f"{value:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.write("### 🟨 Lasso Regression Results")
            for metric, value in lasso_metrics.items():
                st.metric(metric, f"{value:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Comparison
        st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
        st.write("### 🏆 Model Comparison")
        comparison_df = pd.DataFrame({
            'Metric': list(ridge_metrics.keys()),
            'Ridge': list(ridge_metrics.values()),
            'Lasso': list(lasso_metrics.values())
        })
        comparison_df['Better Model'] = comparison_df.apply(
            lambda row: 'Ridge' if (row['Ridge'] > row['Lasso'] and row['Metric'] in ['R²', 'Adj R²']) 
                        or (row['Ridge'] < row['Lasso'] and row['Metric'] in ['MAE', 'MSE', 'RMSE'])
                        else 'Lasso', axis=1
        )
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge Predictions", "Lasso Predictions"))
        
        fig.add_trace(
            go.Scatter(x=y_test, y=ridge_pred, mode='markers', name='Ridge', 
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=y_test, y=lasso_pred, mode='markers', name='Lasso',
                      marker=dict(color='orange', opacity=0.6)),
            row=1, col=2
        )
        
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), ridge_pred.min(), lasso_pred.min()), max(y_test.max(), ridge_pred.max(), lasso_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                      name='Perfect Prediction', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                      name='Perfect Prediction', line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Values")
        fig.update_yaxes(title_text="Predicted Values")
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save to MongoDB
        if st.button("💾 Save Ridge & Lasso Results"):
            success_ridge = save_to_mongodb("Ridge", dataset_choice, ridge_metrics, {"alpha": ridge_alpha})
            success_lasso = save_to_mongodb("Lasso", dataset_choice, lasso_metrics, {"alpha": lasso_alpha})
            if success_ridge and success_lasso:
                st.success("✅ Results saved to MongoDB!")
    
    if model_choice in ["Polynomial Regression", "All Models"]:
        st.subheader("🔄 Polynomial Regression")
        
        degree = st.slider("Polynomial Degree", 1, 10, 3)
        
        # Train polynomial model
        poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        poly_model.fit(X_train, y_train)
        
        # Predictions
        poly_pred = poly_model.predict(X_test)
        
        # Evaluate model
        poly_metrics = evaluate_model(y_test, poly_pred, X_test.shape[1] * degree)
        
        # Display results
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.write(f"### 🟣 Polynomial Regression (Degree {degree}) Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics_cols = [col1, col2, col3, col4, col5]
        for i, (metric, value) in enumerate(poly_metrics.items()):
            with metrics_cols[i]:
                st.metric(metric, f"{value:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization for polynomial regression
        if len(X.columns) == 1:
            # Create smooth curve for visualization
            X_plot = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 300).reshape(-1, 1)
            y_plot = poly_model.predict(X_plot)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_test.iloc[:, 0], y=y_test, mode='markers', 
                                   name='Test Data', marker=dict(color='blue', size=8)))
            fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_plot, mode='lines', 
                                   name=f'Polynomial Fit (degree {degree})', 
                                   line=dict(color='red', width=3)))
            
            fig.update_layout(
                title=f"Polynomial Regression (Degree {degree})",
                xaxis_title=X.columns[0],
                yaxis_title="Target",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Save to MongoDB
        if st.button("💾 Save Polynomial Results"):
            success = save_to_mongodb("Polynomial", dataset_choice, poly_metrics, {"degree": degree})
            if success:
                st.success("✅ Results saved to MongoDB!")
    
    # Historical Results
    st.header("📚 Historical Results")
    if st.button("🔄 Load Previous Results"):
        results = load_from_mongodb()
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.info("No previous results found.")
    
    # Feature Importance (for Ridge and Lasso)
    if model_choice in ["Ridge & Lasso Comparison", "All Models"] and len(X.columns) > 1:
        st.header("🎯 Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Ridge Coefficients")
            ridge_importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': ridge_model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(ridge_importance, x='Feature', y='Coefficient', 
                        title="Ridge Regression Coefficients")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Lasso Coefficients")
            lasso_importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lasso_model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(lasso_importance, x='Feature', y='Coefficient', 
                        title="Lasso Regression Coefficients")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()