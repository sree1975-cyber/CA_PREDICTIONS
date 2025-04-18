# 1. Required imports and configuration
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

# Configure the app
st.set_page_config(
    page_title="Enhanced Chronic Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stMetric {border: 1px solid #f0f2f6; border-radius: 0.5rem; padding: 1rem;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-medium {color: #ffa500; font-weight: bold;}
    .risk-low {color: #2ecc71; font-weight: bold;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .shap-watermark {display: none !important;}
    div.stPlotlyChart {border: 1px solid #f0f2f6; border-radius: 0.5rem;}
    div.stShap {width: 100% !important; margin: 0 auto !important;}
    div.stShap svg {width: 100% !important; height: auto !important;}
    .stSlider {padding: 0.5rem;}
    .feature-importance-container {width: 100%; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)
# 2. Initialize all session state variables
def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = {}
    if 'citywide_mode' not in st.session_state:
        st.session_state.citywide_mode = False
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
        st.session_state.what_if_params = {}
    if 'what_if_changes' not in st.session_state:
        st.session_state.what_if_changes = {}

# Initialize the session state
initialize_session_state()
# 3. Enhanced data preprocessing with proper unknown value handling
def preprocess_data(df, is_training=True):
    """Handle data preprocessing with proper unknown value handling"""
    df = df.copy()
    
    # Calculate attendance percentage if not present
    if 'Attendance_Percentage' not in df.columns:
        if 'Present_Days' in df.columns and 'Absent_Days' in df.columns:
            total_days = df['Present_Days'] + df['Absent_Days']
            df['Attendance_Percentage'] = (df['Present_Days'] / total_days) * 100
    
    # Handle categorical features with proper unknown value handling
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                # Add 'Unknown' category for categorical features during training
                if col == 'Gender':
                    classes = list(df[col].unique()) + ['Unknown']
                    le.fit(classes)
                else:
                    le.fit(df[col])
                df[col] = le.transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                le = st.session_state.label_encoders.get(col)
                if le:
                    # Transform with unknown handling
                    df[col] = df[col].apply(
                        lambda x: x if x in le.classes_ else 'Unknown'
                    )
                    try:
                        df[col] = le.transform(df[col])
                    except ValueError as e:
                        st.error(f"Error encoding {col}: {str(e)}")
                        df[col] = 0  # Default value for unknown categories
    return df
# 4. Model training function with enhanced error handling
def train_model(df):
    """Train ensemble model with enhanced error handling"""
    try:
        df_processed = preprocess_data(df)
        
        # Handle different CA_Status formats
        if df_processed['CA_Status'].dtype == 'object':
            df_processed['CA_Status'] = df_processed['CA_Status'].map({'NO_CA': 0, 'CA': 1}).astype(int)
        elif df_processed['CA_Status'].dtype == 'bool':
            df_processed['CA_Status'] = df_processed['CA_Status'].astype(int)
        
        # Validate target variable
        unique_values = df_processed['CA_Status'].unique()
        if set(unique_values) != {0, 1}:
            st.error(f"Target variable must be binary (0/1). Found values: {unique_values}")
            return None, None, None
        
        # Prepare features and target
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize models
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Create ensemble model
        model = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf)],
            voting='soft'
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Generate predictions and report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Prepare SHAP explainer
        explainer = shap.TreeExplainer(model.named_estimators_['xgb'])
        shap_values = explainer.shap_values(X_train)
        
        return model, report, (explainer, shap_values, X_train)
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None
# 5. Enhanced feature importance visualization
def plot_feature_importance(model):
    """Create interactive feature importance plot using Plotly"""
    try:
        if hasattr(model, 'named_estimators_'):
            xgb_model = model.named_estimators_['xgb']
            
            # Get feature importance
            importance = xgb_model.feature_importances_
            features = xgb_model.feature_names_in_
            
            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Create interactive bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                height=500,
                color='Importance',
                color_continuous_scale='Bluered'
            )
            
            # Update layout for better readability
            fig.update_layout(
                margin=dict(l=100, r=50, t=80, b=50),
                xaxis_title="Importance Score",
                yaxis_title="Features",
                yaxis={'categoryorder':'total ascending'},
                hovermode="y"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Could not generate feature importance plot: {str(e)}")
# 6. Enhanced What-If Analysis without page refresh
def what_if_analysis(student_data):
    """Interactive what-if analysis with session state management"""
    st.subheader("What-If Analysis")
    
    # Initialize parameters in session state if not present
    if 'what_if_params' not in st.session_state:
        st.session_state.what_if_params = {
            'present': student_data.get('Present_Days', 90),
            'absent': student_data.get('Absent_Days', 10),
            'performance': student_data.get('Academic_Performance', 75)
        }
    
    # Create columns for sliders
    col1, col2 = st.columns(2)
    
    with col1:
        # Present days slider
        st.session_state.what_if_params['present'] = st.slider(
            "Present Days", 
            min_value=0, 
            max_value=200,
            value=st.session_state.what_if_params['present'],
            key="wi_present"
        )
        
    with col2:
        # Absent days slider
        st.session_state.what_if_params['absent'] = st.slider(
            "Absent Days",
            min_value=0,
            max_value=200,
            value=st.session_state.what_if_params['absent'],
            key="wi_absent"
        )
    
    # Academic performance slider
    st.session_state.what_if_params['performance'] = st.slider(
        "Academic Performance",
        min_value=0,
        max_value=100,
        value=st.session_state.what_if_params['performance'],
        key="wi_performance"
    )
    
    # Calculate button
    if st.button("Calculate New Risk", key="wi_calculate"):
        # Create modified data
        modified_data = student_data.copy()
        modified_data['Present_Days'] = st.session_state.what_if_params['present']
        modified_data['Absent_Days'] = st.session_state.what_if_params['absent']
        modified_data['Academic_Performance'] = st.session_state.what_if_params['performance']
        
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
        
        # Store changes
        st.session_state.what_if_changes = {
            'original': original_risk,
            'new': new_risk,
            'change': new_risk - original_risk
        }
# 7. Enhanced Single Student Check with all fixes
def single_student_check():
    """Single student analysis with all requested fixes"""
    st.header("👤 Single Student Check")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
        return
    
    # Input form
    with st.form(key='student_input_form'):
        st.subheader("Student Information")
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID (Optional)", key="student_id")
            grade = st.selectbox("Grade", range(1, 13), index=5, key="grade")
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"], key="gender")
            meal_code = st.selectbox("Meal Code", ["Free", "Reduced", "Paid", "Unknown"], key="meal_code")
        
        with col2:
            present_days = st.number_input("Present Days", min_value=0, max_value=365, value=45, key="present_days")
            absent_days = st.number_input("Absent Days", min_value=0, max_value=365, value=10, key="absent_days")
            academic_performance = st.number_input("Academic Performance (0-100)", 
                                                min_value=0, max_value=100, value=75, key="academic_performance")
            
            if st.session_state.citywide_mode:
                transferred = st.checkbox("Transferred student?", key="transferred")
                if transferred:
                    prev_ca = st.selectbox("Previous school CA status", 
                                         ["Unknown", "Yes", "No"], key="prev_ca")
        
        submitted = st.form_submit_button("Check Risk", type="primary")
    
    # Results display
    if submitted:
        input_data = {
            'Student_ID': student_id,
            'Grade': grade,
            'Gender': gender,
            'Present_Days': present_days,
            'Absent_Days': absent_days,
            'Meal_Code': meal_code,
            'Academic_Performance': academic_performance
        }
        
        attendance_pct = (present_days / (present_days + absent_days)) * 100
        risk = predict_ca_risk(input_data, st.session_state.model)
        
        if risk is not None:
            risk = float(risk[0])
            
            # Adjust risk for transferred students
            if st.session_state.citywide_mode and transferred and prev_ca == "Yes":
                risk = min(risk * 1.4, 0.99)
            
            # Determine risk level
            thresholds = st.session_state.risk_thresholds
            if risk < thresholds['low']:
                risk_level = "Low"
                risk_class = "risk-low"
            elif risk < thresholds['medium']:
                risk_level = "Medium"
                risk_class = "risk-medium"
            else:
                risk_level = "High"
                risk_class = "risk-high"
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stMetric">
                    <h3>CA Risk Level</h3>
                    <p class="{risk_class}" style="font-size: 2rem;">{risk_level}</p>
                    <p>Probability: {risk:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stMetric">
                    <h3>Attendance</h3>
                    <p style="font-size: 2rem;">{attendance_pct:.1f}%</p>
                    <p>{present_days} present / {absent_days} absent days</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CA Risk Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, thresholds['low']*100], 'color': "lightgreen"},
                        {'range': [thresholds['low']*100, thresholds['medium']*100], 'color': "orange"},
                        {'range': [thresholds['medium']*100, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk * 100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance visualization
            st.subheader("Feature Importance")
            plot_feature_importance(st.session_state.model)
            
            # Historical trends
            if student_id and student_id in st.session_state.student_history:
                st.subheader("Historical Trends")
                plot_student_history(student_id)
            
            # What-if analysis
            what_if_analysis(input_data)
            
            # Recommendations
            st.subheader("Recommended Actions")
            if risk_level == "High":
                st.markdown("""
                - **Immediate counselor meeting**
                - **Parent/guardian notification**
                - **Attendance improvement plan**
                - **Academic support services**
                - **Weekly monitoring**
                """)
            elif risk_level == "Medium":
                st.markdown("""
                - **Monthly check-ins**
                - **Mentor assignment**
                - **After-school program referral**
                - **Quarterly parent meetings**
                """)
            else:
                st.markdown("""
                - **Continue regular monitoring**
                - **Positive reinforcement**
                - **Encourage extracurriculars**
                """)
# 8. Main application structure
def main():
    st.title("🏫 Enhanced Chronic Absenteeism Early Warning System")
    st.markdown("Predict students at risk of chronic absenteeism (CA) using advanced analytics and machine learning.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", 
                              ["System Training", 
                               "Batch Prediction", 
                               "Single Student Check",
                               "Advanced Analytics",
                               "System Settings"])
    
    # Route to appropriate section
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
