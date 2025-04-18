# chronic_absenteeism_predictor.py
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
import tempfile
import base64
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

# ========================
# 1. APP CONFIGURATION
# ========================
st.set_page_config(
    page_title="Chronic Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)

# ========================
# 2. SESSION STATE SETUP
# ========================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'current_df' not in st.session_state:
    st.session_state.current_df = pd.DataFrame()
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'risk_thresholds' not in st.session_state:
    st.session_state.risk_thresholds = {'low': 0.3, 'medium': 0.7, 'high': 1.0}

# ========================
# 3. CORE FUNCTIONALITY
# ========================

def generate_sample_data():
    """Generate comprehensive sample dataset"""
    np.random.seed(42)
    num_students = 500
    date_ranges = pd.date_range(start='2019-09-01', end='2022-06-30', freq='D')
    
    historical_data = pd.DataFrame({
        'Student_ID': [f'STD{1000+i}' for i in range(num_students)] * 3,
        'Date': np.random.choice(date_ranges, num_students*3),
        'Present_Days': np.random.randint(80, 180, num_students*3),
        'Absent_Days': np.random.randint(0, 30, num_students*3),
        'Academic_Performance': np.random.randint(50, 100, num_students*3),
        'CA_Status': np.random.choice([0, 1], num_students*3, p=[0.8, 0.2])
    })
    
    historical_data['Attendance_Percentage'] = (
        historical_data['Present_Days'] / 
        (historical_data['Present_Days'] + historical_data['Absent_Days'])
    ) * 100
    
    return historical_data

def preprocess_data(df, is_training=True):
    """Full data preprocessing pipeline"""
    df = df.copy()
    
    # Date handling
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Feature engineering
    if all(col in df.columns for col in ['Present_Days', 'Absent_Days']):
        df['Attendance_Percentage'] = (
            df['Present_Days'] / 
            (df['Present_Days'] + df['Absent_Days'])
        ) * 100
    
    # Categorical encoding
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                classes = list(df[col].unique()) + ['Unknown']
                le.fit(classes)
                df[col] = le.transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                le = st.session_state.label_encoders.get(col)
                if le:
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    df[col] = le.transform(df[col])
    return df

def train_model(df):
    """Complete model training workflow"""
    try:
        df_processed = preprocess_data(df)
        
        if 'CA_Status' not in df_processed.columns:
            st.error("Target variable 'CA_Status' missing")
            return None, None
        
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=5))
            ],
            voting='soft'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, report
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None

# ========================
# 4. VISUALIZATIONS
# ========================

def plot_temporal_trends():
    """Fixed temporal trends visualization"""
    if not st.session_state.historical_data.empty:
        df = st.session_state.historical_data.copy()
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        
        monthly_avg = df.groupby('Month').agg({
            'Attendance_Percentage': 'mean',
            'CA_Status': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_avg['Month'],
            y=monthly_avg['Attendance_Percentage'],
            name='Attendance %',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_avg['Month'],
            y=monthly_avg['CA_Status']*100,
            name='CA Rate %',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Historical Attendance Trends',
            yaxis=dict(title='Attendance Percentage'),
            yaxis2=dict(title='CA Rate Percentage', overlaying='y', side='right'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload historical data to view temporal trends")

def what_if_analysis(student_data):
    """Persistent What-If analysis with state management"""
    st.subheader("Scenario Analysis")
    
    # Initialize session state
    if 'what_if_params' not in st.session_state:
        st.session_state.what_if_params = {
            'present': student_data['Present_Days'],
            'absent': student_data['Absent_Days'],
            'performance': student_data['Academic_Performance']
        }
    
    # Interactive controls
    col1, col2 = st.columns(2)
    with col1:
        present = st.slider(
            "Present Days", 
            min_value=0, 
            max_value=200,
            value=st.session_state.what_if_params['present'],
            key='wi_present'
        )
    with col2:
        absent = st.slider(
            "Absent Days",
            min_value=0,
            max_value=200,
            value=st.session_state.what_if_params['absent'],
            key='wi_absent'
        )
    
    performance = st.slider(
        "Academic Performance",
        min_value=0,
        max_value=100,
        value=st.session_state.what_if_params['performance'],
        key='wi_performance'
    )
    
    if st.button("Calculate New Risk"):
        # Update session state
        st.session_state.what_if_params.update({
            'present': present,
            'absent': absent,
            'performance': performance
        })
        
        # Calculate risks
        modified_data = student_data.copy()
        modified_data.update({
            'Present_Days': present,
            'Absent_Days': absent,
            'Academic_Performance': performance
        })
        
        original_risk = st.session_state.model.predict_proba(
            preprocess_data(pd.DataFrame([student_data]), False)
        )[0][1]
        
        new_risk = st.session_state.model.predict_proba(
            preprocess_data(pd.DataFrame([modified_data]), False)
        )[0][1]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Risk", f"{original_risk:.1%}")
        with col2:
            st.metric("New Risk", f"{new_risk:.1%}", 
                     delta=f"{new_risk - original_risk:+.1%}")

# ========================
# 5. MAIN APPLICATION UI
# ========================

def main():
    """Main application flow"""
    st.title("Chronic Absenteeism Prediction System")
    
    menu = st.sidebar.selectbox("Navigation", [
        "System Training",
        "Batch Prediction",
        "Single Student Analysis",
        "Advanced Analytics"
    ])
    
    if menu == "System Training":
        st.header("Model Training")
        uploaded_file = st.file_uploader("Upload historical data", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.historical_data = df
            
            if st.button("Train Model"):
                model, report = train_model(df)
                if model:
                    st.session_state.model = model
                    st.success("Model trained successfully!")
                    st.json(report)
    
    elif menu == "Single Student Analysis":
        st.header("Individual Student Risk Assessment")
        
        with st.form("student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.text_input("Student ID")
                grade = st.selectbox("Grade", range(1, 13))
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
            
            with col2:
                present_days = st.number_input("Present Days", min_value=0, value=45)
                absent_days = st.number_input("Absent Days", min_value=0, value=10)
                academic_performance = st.number_input("Academic Performance", 0, 100, 75)
            
            if st.form_submit_button("Calculate Risk"):
                input_data = {
                    'Student_ID': student_id,
                    'Grade': grade,
                    'Gender': gender,
                    'Present_Days': present_days,
                    'Absent_Days': absent_days,
                    'Academic_Performance': academic_performance
                }
                
                if st.session_state.model:
                    what_if_analysis(input_data)
                else:
                    st.warning("Please train a model first")
    
    elif menu == "Advanced Analytics":
        st.header("Advanced Analytics Dashboard")
        plot_temporal_trends()
        
        if not st.session_state.historical_data.empty:
            st.subheader("Geographic Distribution")
            generate_geographic_map(st.session_state.historical_data)

if __name__ == "__main__":
    main()
