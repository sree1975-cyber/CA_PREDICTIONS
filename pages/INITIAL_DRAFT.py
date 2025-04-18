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

# Model training functions
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    return model, scaler

def train_lstm(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    
    model = Sequential([
        LSTM(64, input_shape=(1, X_train_reshaped.shape[2])),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
    return model, scaler

# Evaluation function
def evaluate_model(model, X_test, y_test, model_type='standard'):
    if model_type in ['nn', 'lstm']:
        model, scaler = model
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'lstm':
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

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
                rf_model = train_random_forest(X_train, y_train)
                rf_results = evaluate_model(rf_model, X_test, y_test)
                results["Random Forest"] = rf_results
                models["Random Forest"] = rf_model
                
            if "XGBoost" in models_to_train:
                xgb_model = train_xgboost(X_train, y_train)
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
                st.table(pd.DataFrame(result['report']).transpose())
                
                st.subheader(f"Confusion Matrix for {model_name}")
                fig = px.imshow(result['confusion_matrix'],
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Not Chronic', 'Chronic'],
                               y=['Not Chronic', 'Chronic'],
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for tree-based models
        if "Random Forest" in st.session_state.models:
            st.subheader("Feature Importance (Random Forest)")
            rf_model = st.session_state.models["Random Forest"]
            importances = rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='Importance', y='Feature',
                         title='Top 10 Important Features (Random Forest)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model saving
        st.subheader("Model Persistence")
        selected_model = st.selectbox("Select model to save", list(st.session_state.models.keys()))
        
        if st.button("Save Selected Model"):
            model_to_save = st.session_state.models[selected_model]
            model_filename = f"{selected_model.replace(' ', '_').lower()}_absenteeism_model.pkl"
            
            if selected_model in ["Neural Network", "LSTM"]:
                # For Keras models, we need to save both model and scaler
                with open(model_filename, 'wb') as f:
                    pickle.dump(model_to_save, f)
            else:
                joblib.dump(model_to_save, model_filename)
            
            with open(model_filename, 'rb') as f:
                st.download_button(
                    label="Download Model",
                    data=f,
                    file_name=model_filename,
                    mime='application/octet-stream'
                )
            
            os.remove(model_filename)
            st.success(f"{selected_model} saved successfully!")

elif app_mode == "Predictions":
    st.header("Make Predictions")
    
    # Check if models are available
    if 'models' not in st.session_state:
        st.warning("Please train models in the 'Model Training & Evaluation' section first.")
        st.stop()
    
    # Model selection for predictions
    selected_model = st.selectbox("Select model for predictions", 
                                list(st.session_state.models.keys()))
    
    # Input form for prediction
    st.subheader("Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        socioeconomic_status = st.selectbox("Socioeconomic Status", 
                                         ["Low", "Medium", "High", "Unknown"])
        academic_performance = st.number_input("Academic Performance (0-100)", 
                                            min_value=0, max_value=100, value=70)
    
    with col2:
        ytd_attendance = st.number_input("YTD Attendance (%)", 
                                       min_value=0, max_value=100, value=85)
        gender = st.selectbox("Gender", 
                            ["Male", "Female", "Other", "Unknown"])
    
    # Handle dynamic features
    dynamic_features = {}
    for col in st.session_state.df.columns:
        if col.startswith('Extended_Feature_'):
            dynamic_features[col] = st.number_input(col, value=0.0)
    
    # Prepare input data
    input_data = {
        'Academic_Performance': academic_performance,
        'YTD_Attendance': ytd_attendance,
        'Socioeconomic_Status_High': 1 if socioeconomic_status == 'High' else 0,
        'Socioeconomic_Status_Low': 1 if socioeconomic_status == 'Low' else 0,
        'Socioeconomic_Status_Medium': 1 if socioeconomic_status == 'Medium' else 0,
        'Socioeconomic_Status_Unknown': 1 if socioeconomic_status == 'Unknown' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Other': 1 if gender == 'Other' else 0,
        'Gender_Unknown': 1 if gender == 'Unknown' else 0
    }
    
    # Add dynamic features
    input_data.update(dynamic_features)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns are present (fill missing with 0)
    if "X_train_columns" not in st.session_state:
        st.warning("Please train a model first to generate column structure.")
        st.stop()
    for col in st.session_state.X_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[st.session_state.X_train_columns]
    
    # Make prediction
    if st.button("Predict"):
        model = st.session_state.models[selected_model]
        
        if selected_model in ["Neural Network", "LSTM"]:
            model, scaler = model
            input_scaled = scaler.transform(input_df)
            
            if selected_model == "LSTM":
                input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
            
            prediction = (model.predict(input_scaled) > 0.5).astype(int)[0][0]
        else:
            prediction = model.predict(input_df)[0]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("High Risk of Chronic Absenteeism")
            st.markdown("""
            **Recommended Interventions:**
            - Early warning system alert
            - Counselor follow-up
            - Parent/guardian notification
            - Academic support services
            """)
        else:
            st.success("Low Risk of Chronic Absenteeism")
            st.markdown("""
            **Maintenance Strategies:**
            - Continue monitoring attendance
            - Positive reinforcement
            - Maintain current support systems
            """)
        
        # Show prediction probability if available
        if selected_model in ["Random Forest", "XGBoost"]:
            proba = model.predict_proba(input_df)[0]
            fig = px.bar(x=['Low Risk', 'High Risk'], y=proba,
                         labels={'x': 'Risk Level', 'y': 'Probability'},
                         title='Prediction Probability')
            st.plotly_chart(fig, use_container_width=True)

elif app_mode == "API Integration":
    st.header("API Integration")
    
    st.markdown("""
    ## How to integrate with external applications
    
    You can integrate this model with external applications using a REST API.
    Below is sample code for creating and consuming the API.
    """)
    
    # API creation example
    st.subheader("API Creation (Flask Example)")
    st.code("""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_absenteeism_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()
        
        # Return response
        return jsonify({
            'prediction': int(prediction),
            'probability': probability,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    """, language='python')
    
    # API consumption example
    st.subheader("API Consumption Example")
    st.code("""
import requests
import json

# API endpoint
url = 'http://your-api-url/predict'

# Sample data
data = {
    'Academic_Performance': 75,
    'YTD_Attendance': 80,
    'Socioeconomic_Status_High': 0,
    'Socioeconomic_Status_Low': 1,
    'Socioeconomic_Status_Medium': 0,
    'Gender_Female': 0,
    'Gender_Male': 1,
    'Extended_Feature_1': 0.5
}

# Make request
response = requests.post(url, json=data)

# Process response
if response.status_code == 200:
    result = response.json()
    if result['status'] == 'success':
        print(f"Prediction: {'High Risk' if result['prediction'] else 'Low Risk'}")
        print(f"Probabilities: {result['probability']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
else:
    print(f"Request failed with status code {response.status_code}")
    """, language='python')
    
    # FastAPI alternative
    st.subheader("FastAPI Alternative")
    st.markdown("""
    For better performance and async support, consider using FastAPI:
    """)
    st.code("""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class StudentData(BaseModel):
    Academic_Performance: float
    YTD_Attendance: float
    Socioeconomic_Status_High: int
    Socioeconomic_Status_Low: int
    Socioeconomic_Status_Medium: int
    Gender_Female: int
    Gender_Male: int
    Extended_Feature_1: float = 0.0

# Load model
model = joblib.load('random_forest_absenteeism_model.pkl')

@app.post("/predict")
async def predict(data: StudentData):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()
        
        return {
            'prediction': int(prediction),
            'probability': probability,
            'status': 'success'
        }
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}
    """, language='python')

# Store data in session state
if 'df' in locals():
    st.session_state.df = df
if 'X_train_columns' in locals():
    st.session_state.X_train_columns = X_train.columns.tolist()
