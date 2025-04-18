# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import joblib
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import os
from io import BytesIO
import requests
from PIL import Image

# Set up Streamlit app
st.set_page_config(page_title="Chronic Absenteeism Predictor", layout="wide")

# App title and description
st.title("Chronic Absenteeism Prediction System")
st.markdown("""
This tool predicts chronic absenteeism risk based on student characteristics.
Upload your Excel data or use our sample dataset to get started.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                               ["Data Upload & Exploration", 
                                "Model Training & Evaluation",
                                "Predictions",
                                "API Integration"])

# Sample data generation function
def generate_sample_data():
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Student_ID': range(1, n_samples+1),
        'Socioeconomic_Status': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.3, 0.3]),
        'Academic_Performance': np.random.normal(70, 15, n_samples).clip(0, 100),
        'YTD_Attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'Chronic_Absenteeism': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # Add some dynamic columns
    for i in range(1, 4):
        data[f'Extended_Feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)

# Data preprocessing function
def preprocess_data(df):
    # Handle missing values
    df.fillna({
        'Socioeconomic_Status': 'Unknown',
        'Academic_Performance': df['Academic_Performance'].median(),
        'YTD_Attendance': df['YTD_Attendance'].median(),
        'Gender': 'Unknown'
    }, inplace=True)
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Socioeconomic_Status', 'Gender'])
    
    # Ensure all expected columns are present
    expected_cols = ['Academic_Performance', 'YTD_Attendance',
                    'Socioeconomic_Status_High', 'Socioeconomic_Status_Low',
                    'Socioeconomic_Status_Medium', 'Socioeconomic_Status_Unknown',
                    'Gender_Female', 'Gender_Male', 'Gender_Other', 'Gender_Unknown']
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df

# Main app logic
if app_mode == "Data Upload & Exploration":
    st.header("Data Upload & Exploration")
    
    # Data upload options
    data_option = st.radio("Choose data source:", 
                          ("Upload Excel file", "Use sample data"))
    
    if data_option == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df = None
        else:
            df = None
    else:
        df = generate_sample_data()
        st.info("Using sample data. You can download this to understand the expected format.")
        st.download_button(
            label="Download Sample Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='sample_absenteeism_data.csv',
            mime='text/csv'
        )
    
    if df is not None:
        # Data exploration
        st.subheader("Data Preview")
        st.write(df.head())
        
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        st.subheader("Data Visualizations")
        
        # Plot attendance distribution
        fig1 = px.histogram(df, x='YTD_Attendance', 
                           title='Year-to-Date Attendance Distribution',
                           color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot absenteeism by socioeconomic status
        if 'Socioeconomic_Status' in df.columns:
            fig2 = px.box(df, x='Socioeconomic_Status', y='YTD_Attendance',
                         title='Attendance by Socioeconomic Status')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Plot academic performance vs attendance
        fig3 = px.scatter(df, x='Academic_Performance', y='YTD_Attendance',
                         color='Chronic_Absenteeism' if 'Chronic_Absenteeism' in df.columns else None,
                         title='Academic Performance vs Attendance')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Show missing values
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        st.write(missing_data)
        
        # Allow user to add dynamic columns
        st.subheader("Add Dynamic Columns")
        num_new_cols = st.number_input("Number of new columns to add", min_value=0, max_value=10, value=0)
        
        for i in range(num_new_cols):
            col_name = st.text_input(f"Name for new column {i+1}", value=f"Extended_Feature_{i+1}")
            if col_name:
                df[col_name] = np.random.normal(0, 1, len(df))  # Sample random data
    
elif app_mode == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    # Check if data is available
    if 'df' not in st.session_state:
        st.warning("Please upload or generate data in the 'Data Upload & Exploration' section first.")
        st.stop()
    
    df = st.session_state.df
    
    # Check if target variable exists
    if 'Chronic_Absenteeism' not in df.columns:
        st.error("Target variable 'Chronic_Absenteeism' not found in data.")
        st.stop()
    
    # Preprocess data
    df_processed = preprocess_data(df.copy())
    
    # Split data
    X = df_processed.drop('Chronic_Absenteeism', axis=1)
    y = df_processed['Chronic_Absenteeism']
    
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    st.session_state.X_train_columns = X_train.columns.tolist()
    
    # Model selection
    st.subheader("Model Selection")
    models_to_train = st.multiselect("Select models to train", 
                                    ["Random Forest", "XGBoost", "Neural Network", "LSTM"],
                                    default=["Random Forest", "XGBoost"])
    
    # Train models
    if st.button("Train Selected Models"):
        results = {}
        models = {}
        
        with st.spinner("Training models..."):
            if "Random Forest" in models_to_train:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_results = evaluate_model(rf_model, X_test, y_test)
                results["Random Forest"] = rf_results
                models["Random Forest"] = rf_model
                
            if "XGBoost" in models_to_train:
                xgb_model = XGBClassifier(n_estimators=100, random_state=42)
                xgb_model.fit(X_train, y_train)
                xgb_results = evaluate_model(xgb_model, X_test, y_test)
                results["XGBoost"] = xgb_results
                models["XGBoost"] = xgb_model
                
            if "Neural Network" in models_to_train:
                nn_model, nn_scaler = train_neural_network(X_train, y_train)
                nn_results = evaluate_model((nn_model, nn_scaler), X_test, y_test, 'nn')
                results["Neural Network"] = nn_results
                models["Neural Network"] = (nn_model, nn_scaler)
                
            if "LSTM" in models_to_train:
                lstm_model, lstm_scaler = train_lstm(X_train, y_train)
                lstm_results = evaluate_model((lstm_model, lstm_scaler), X_test, y_test, 'lstm')
                results["LSTM"] = lstm_results
                models["LSTM"] = (lstm_model, lstm_scaler)
        
        st.session_state.results = results
        st.session_state.models = models
        st.success("Model training completed!")
    
    # Display results if available
    if 'results' in st.session_state:
        st.subheader("Model Evaluation Results")
        
        # Create a comparison table
        comparison_data = []
        for model_name, result in st.session_state.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision (Class 1)': result['report']['1']['precision'],
                'Recall (Class 1)': result['report']['1']['recall'],
                'F1-Score (Class 1)': result['report']['1']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.highlight_max(axis=0))
        
        # Show detailed results for each model
        for model_name, result in st.session_state.results.items():
            with st.expander(f"Detailed Results for {model_name}"):
                st.subheader(f"Classification Report for {model_name}")
                st.write(result['report'])
                st.subheader(f"Confusion Matrix for {model_name}")
                st.write(result['conf_matrix'])

elif app_mode == "Predictions":
    st.header("Make Predictions")
    
    if 'df' not in st.session_state:
        st.warning("Please upload or generate data in the 'Data Upload & Exploration' section first.")
        st.stop()
    
    df = st.session_state.df
    
    if 'models' not in st.session_state:
        st.warning("Please train models in the 'Model Training & Evaluation' section first.")
        st.stop()
    
    model_name = st.selectbox("Choose a model for prediction:", list(st.session_state.models.keys()))
    model = st.session_state.models[model_name]
    
    if model_name in ["Neural Network", "LSTM"]:
        nn_scaler = model[1]
        model = model[0]
        
    # Preprocess the input data
    df_processed = preprocess_data(df.copy())
    X_new = df_processed.drop('Chronic_Absenteeism', axis=1)
    
    # Predict using the selected model
    st.subheader(f"Predictions with {model_name}")
    predictions = model.predict(X_new)
    
    df['Predicted_Absenteeism'] = predictions
    st.write(df[['Student_ID', 'Predicted_Absenteeism']])

else:
    st.warning("Invalid mode selected.")
