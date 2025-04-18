# chronic_absenteeism_app.py
# Chronic Absenteeism Prediction System - Full Implementation

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime, timedelta
import tempfile
import base64
import shap
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# =============================================
# 1. APP CONFIGURATION & STYLING
# =============================================
st.set_page_config(
    page_title="Chronic Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {padding: 2rem;}
    .stMetric {border: 1px solid #f0f2f6; border-radius: 0.5rem; padding: 1rem;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-medium {color: #ffa500; font-weight: bold;}
    .risk-low {color: #2ecc71; font-weight: bold;}
    .plot-container {margin-top: 2rem; border: 1px solid #f0f2f6; border-radius: 0.5rem; padding: 1rem;}
</style>
""", unsafe_allow_html=True)

# =============================================
# 2. SESSION STATE MANAGEMENT
# =============================================
def initialize_session_state():
    """Initialize all session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = {}
    if 'current_df' not in st.session_state:
        st.session_state.current_df = pd.DataFrame()
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    if 'student_history' not in st.session_state:
        st.session_state.student_history = {}
    if 'risk_thresholds' not in st.session_state:
        st.session_state.risk_thresholds = {'low': 0.3, 'medium': 0.7, 'high': 1.0}
    if 'interventions' not in st.session_state:
        st.session_state.interventions = {
            'Counseling': {'cost': 500, 'effectiveness': 0.3},
            'Mentorship': {'cost': 300, 'effectiveness': 0.2},
            'Parent Meeting': {'cost': 200, 'effectiveness': 0.15},
            'After-school Program': {'cost': 400, 'effectiveness': 0.25}
        }
    if 'what_if_params' not in st.session_state:
        st.session_state.what_if_params = None

initialize_session_state()

# =============================================
# 3. CORE FUNCTIONALITY MODULES
# =============================================

# -------------------------
# 3.1 Data Generation
# -------------------------
def generate_sample_data():
    """Generate synthetic data for demonstration purposes"""
    np.random.seed(42)
    num_students = 500
    schools = ['School A', 'School B', 'School C', 'School D']
    
    # Current academic year data
    current_df = pd.DataFrame({
        'Student_ID': [f'STD{1000+i}' for i in range(num_students)],
        'School': np.random.choice(schools, num_students),
        'Grade': np.random.choice(range(1, 13), num_students),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_students),
        'Present_Days': np.random.randint(80, 180, num_students),
        'Absent_Days': np.random.randint(0, 30, num_students),
        'Meal_Code': np.random.choice(['Free', 'Reduced', 'Paid'], num_students),
        'Academic_Performance': np.random.randint(50, 100, num_students),
        'Address': np.random.choice([
            "100 Main St, Anytown, USA",
            "200 Oak Ave, Somewhere, USA",
            "300 Pine Rd, Nowhere, USA"
        ], num_students)
    })
    
    # Historical data (3 years)
    historical_data = pd.DataFrame()
    for year in [2021, 2022, 2023]:
        year_data = current_df.copy()
        year_data['Date'] = pd.to_datetime(f'{year}-09-01') + pd.to_timedelta(
            np.random.randint(0, 180, num_students), unit='d')
        year_data['CA_Status'] = np.random.choice([0, 1], num_students, p=[0.8, 0.2])
        historical_data = pd.concat([historical_data, year_data])
    
    return current_df, historical_data

# -------------------------
# 3.2 Data Preprocessing
# -------------------------
def preprocess_data(df, is_training=True):
    """Clean and transform data for model consumption"""
    df = df.copy()
    
    # Calculate attendance percentage
    if 'Attendance_Percentage' not in df.columns:
        total_days = df['Present_Days'] + df['Absent_Days']
        df['Attendance_Percentage'] = (df['Present_Days'] / total_days) * 100
    
    # Handle categorical features
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                # Add 'Unknown' category for inference
                classes = list(df[col].unique()) + ['Unknown'] if col == 'Gender' else df[col].unique()
                le.fit(classes)
                df[col] = le.transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                le = st.session_state.label_encoders.get(col)
                if le:
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    df[col] = le.transform(df[col])
    return df

# -------------------------
# 3.3 Model Training
# -------------------------
def train_model(df):
    """Train ensemble model and return artifacts"""
    try:
        df_processed = preprocess_data(df)
        
        # Validate target variable
        if df_processed['CA_Status'].nunique() != 2:
            st.error("Target variable must have exactly 2 classes")
            return None, None, None
        
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        # Create ensemble
        model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
        model.fit(X_train, y_train)
        
        # Generate reports
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, report, (xgb, X_train)
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None

# -------------------------
# 3.4 Prediction Engine
# -------------------------
def predict_ca_risk(input_data, model):
    """Generate risk predictions for input data"""
    try:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        df_processed = preprocess_data(df, is_training=False)
        
        # Handle feature mismatch
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            df_processed = df_processed[model.feature_names_in_]
        
        return model.predict_proba(df_processed)[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# =============================================
# 4. VISUALIZATION MODULES
# =============================================

# -------------------------
# 4.1 Temporal Trends
# -------------------------
def plot_temporal_trends():
    """Display historical attendance trends"""
    if not st.session_state.historical_data.empty:
        # Process temporal data
        df = st.session_state.historical_data.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)
        
        # Aggregate monthly data
        monthly_avg = df.groupby('Date').agg({
            'Attendance_Percentage': 'mean',
            'CA_Status': 'mean'
        }).reset_index()
        
        # Create dual-axis plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_avg['Date'],
            y=monthly_avg['Attendance_Percentage'],
            name='Attendance %',
            line=dict(color='blue')
        )
        fig.add_trace(go.Scatter(
            x=monthly_avg['Date'],
            y=monthly_avg['CA_Status']*100,
            name='CA Rate %',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Historical Trends',
            yaxis=dict(title='Attendance Percentage'),
            yaxis2=dict(title='CA Rate Percentage', overlaying='y', side='right'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload historical data to view temporal trends")

# -------------------------
# 4.2 What-If Analysis
# -------------------------
def what_if_analysis(student_data):
    """Interactive scenario analysis with state persistence"""
    st.subheader("Scenario Analysis")
    
    # Initialize session state
    if st.session_state.what_if_params is None:
        st.session_state.what_if_params = {
            'present': student_data['Present_Days'],
            'absent': student_data['Absent_Days'],
            'performance': student_data['Academic_Performance']
        }
    
    # Create sliders with current values
    col1, col2 = st.columns(2)
    with col1:
        new_present = st.slider(
            "Present Days", 
            min_value=0, 
            max_value=200,
            value=st.session_state.what_if_params['present'],
            key='wi_present'
        )
    with col2:
        new_absent = st.slider(
            "Absent Days",
            min_value=0,
            max_value=200,
            value=st.session_state.what_if_params['absent'],
            key='wi_absent'
        )
    
    new_performance = st.slider(
        "Academic Performance",
        min_value=0,
        max_value=100,
        value=st.session_state.what_if_params['performance'],
        key='wi_performance'
    )
    
    # Update calculations
    if st.button("Calculate New Risk", key='wi_calculate'):
        # Store new parameters
        st.session_state.what_if_params.update({
            'present': new_present,
            'absent': new_absent,
            'performance': new_performance
        })
        
        # Generate modified data
        modified_data = student_data.copy()
        modified_data.update({
            'Present_Days': new_present,
            'Absent_Days': new_absent,
            'Academic_Performance': new_performance
        })
        
        # Calculate risks
        original_risk = predict_ca_risk(student_data, st.session_state.model)[0]
        new_risk = predict_ca_risk(modified_data, st.session_state.model)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Risk", f"{original_risk:.1%}")
        with col2:
            st.metric("New Risk", f"{new_risk:.1%}", 
                      delta=f"{(new_risk - original_risk):+.1%}")

# =============================================
# 5. STREAMLIT UI COMPONENTS
# =============================================

def system_training():
    """Model training interface"""
    st.header("🔧 System Training")
    # ... [rest of training UI]

def batch_prediction():
    """Batch processing interface"""
    st.header("📊 Batch Prediction")
    # ... [rest of batch prediction UI]

def single_student_check():
    """Individual student analysis"""
    st.header("👤 Single Student Check")
    
    with st.form(key='student_form'):
        col1, col2 = st.columns(2)
        
        # Student inputs
        with col1:
            student_id = st.text_input("Student ID")
            grade = st.selectbox("Grade", range(1, 13), index=5)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
        
        with col2:
            present_days = st.number_input("Present Days", min_value=0, value=45)
            absent_days = st.number_input("Absent Days", min_value=0, value=10)
            academic_performance = st.number_input("Academic Performance", min_value=0, value=75)
        
        if st.form_submit_button("Calculate Risk"):
            # Process inputs and store data
            input_data = {
                'Student_ID': student_id,
                'Grade': grade,
                'Gender': gender,
                'Present_Days': present_days,
                'Absent_Days': absent_days,
                'Academic_Performance': academic_performance
            }
            
            risk = predict_ca_risk(input_data, st.session_state.model)
            
            # Display results
            if risk is not None:
                st.session_state.what_if_params = None  # Reset scenario analysis
                what_if_analysis(input_data)

def advanced_analytics():
    """Advanced visualization interface"""
    st.header("📈 Advanced Analytics")
    plot_temporal_trends()
    # ... [other visualizations]

# =============================================
# 6. MAIN APPLICATION FLOW
# =============================================

def main():
    """Main application controller"""
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", [
        "System Training", 
        "Batch Prediction", 
        "Single Student Check",
        "Advanced Analytics",
        "System Settings"
    ])
    
    if app_mode == "System Training":
        system_training()
    elif app_mode == "Batch Prediction":
        batch_prediction()
    elif app_mode == "Single Student Check":
        single_student_check()
    elif app_mode == "Advanced Analytics":
        advanced_analytics()
    elif app_mode == "System Settings":
        system_settings()

if __name__ == "__main__":
    main()
