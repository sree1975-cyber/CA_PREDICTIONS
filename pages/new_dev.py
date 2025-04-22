# Chronic Absenteeism Predictor - Full Working Version
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
import shap
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

# Initialize session state variables
def initialize_session():
    required_states = {
        'model': None,
        'label_encoders': {},
        'current_df': pd.DataFrame(),
        'historical_data': pd.DataFrame(),
        'risk_thresholds': {'low': 0.3, 'medium': 0.7, 'high': 1.0},
        'X_train_columns': [],
        'interventions': {
            'Counseling': {'cost': 500, 'effectiveness': 0.3},
            'Mentorship': {'cost': 300, 'effectiveness': 0.2},
            'Parent Meeting': {'cost': 200, 'effectiveness': 0.15},
            'After-school Program': {'cost': 400, 'effectiveness': 0.25}
        }
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# Configure the app
st.set_page_config(page_title="CA Predictor", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stMetric {border: 1px solid #f0f2f6; border-radius: 0.5rem; padding: 1rem;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-medium {color: #ffa500; font-weight: bold;}
    .risk-low {color: #2ecc71; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def preprocess_data(df, is_training=True):
    """Preprocess data with proper type handling"""
    df = df.copy()
    
    # Calculate attendance percentage
    if {'Present_Days', 'Absent_Days'}.issubset(df.columns):
        total_days = df['Present_Days'] + df['Absent_Days']
        df['Attendance_Percentage'] = (df['Present_Days'] / total_days).replace(np.inf, 0) * 100
    
    # Encode categorical features
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                df[col] = le.transform(df[col].astype(str))
                st.session_state.label_encoders[col] = le
            else:
                le = st.session_state.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
                    df[col] = le.transform(df[col])
    return df

def train_model(df):
    """Train ensemble model with proper column handling"""
    try:
        df_processed = preprocess_data(df)
        
        # Validate target variable
        if 'CA_Status' not in df_processed.columns:
            st.error("Missing CA_Status column in training data")
            return None, None, None
        
        X = df_processed.drop(['CA_Status', 'Student_ID'], errors='ignore')
        y = df_processed['CA_Status'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train_columns = X_train.columns.tolist()
        
        # Model configuration
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        rf = RandomForestClassifier(n_estimators=50, max_depth=3)
        model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
        model.fit(X_train, y_train)
        
        # Generate SHAP explainer
        explainer = shap.TreeExplainer(model.named_estimators_['xgb'])
        shap_values = explainer.shap_values(X_train)
        
        return model, classification_report(y_test, model.predict(X_test), output_dict=True), (explainer, shap_values, X_train)
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None

def predict_ca_risk(input_data, model):
    """Robust prediction function with type handling"""
    try:
        if isinstance(input_data, (dict, pd.Series)):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        df_processed = preprocess_data(df, is_training=False)
        
        # Ensure feature alignment
        missing_cols = set(st.session_state.X_train_columns) - set(df_processed.columns)
        for col in missing_cols:
            df_processed[col] = 0
        df_processed = df_processed[st.session_state.X_train_columns]
        
        # Generate predictions
        probabilities = model.predict_proba(df_processed)[:, 1]
        return probabilities[0] if len(probabilities) == 1 else probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def batch_prediction():
    """Batch prediction with proper data handling"""
    st.header("📊 Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload student data", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        if st.button("Run Predictions"):
            with st.spinner("Predicting..."):
                df['CA_Risk'] = predict_ca_risk(df, st.session_state.model)
                
                # Apply risk thresholds
                thresholds = st.session_state.risk_thresholds
                df['Risk_Level'] = pd.cut(
                    df['CA_Risk'],
                    bins=[0, thresholds['low'], thresholds['medium'], 1],
                    labels=['Low', 'Medium', 'High']
                )
                
                st.session_state.current_df = df
                st.success(f"Predictions complete for {len(df)} students!")
                
                # Show results
                st.dataframe(df)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

def individual_prediction():
    """Individual prediction with batch integration"""
    st.header("👤 Individual Student Check")
    
    # Batch selection
    if not st.session_state.current_df.empty:
        selected_student = st.selectbox("Choose from batch results", 
                                      ["New Student"] + st.session_state.current_df['Student_ID'].tolist())
        if selected_student != "New Student":
            student_data = st.session_state.current_df.query("Student_ID == @selected_student").iloc[0]
            display_prediction(student_data.to_dict())
            return
    
    # Manual input form
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID")
            grade = st.number_input("Grade", 1, 12)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            present_days = st.number_input("Present Days", 0, 365, 180)
            absent_days = st.number_input("Absent Days", 0, 365, 10)
            meal_code = st.selectbox("Meal Code", ["Free", "Reduced", "Paid"])
        
        if st.form_submit_button("Predict"):
            input_data = {
                'Student_ID': student_id,
                'Grade': grade,
                'Gender': gender,
                'Present_Days': present_days,
                'Absent_Days': absent_days,
                'Meal_Code': meal_code
            }
            display_prediction(input_data)

def display_prediction(data):
    """Display prediction results with thresholds"""
    risk_score = predict_ca_risk(data, st.session_state.model)
    if risk_score is None: return
    
    thresholds = st.session_state.risk_thresholds
    risk_level = 'Low' if risk_score < thresholds['low'] else \
                'Medium' if risk_score < thresholds['medium'] else 'High'
    
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{risk_score:.1%}")
    with col2:
        st.metric("Risk Level", risk_level, 
                 delta_color="inverse" if risk_level != 'Low' else "normal")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'steps': [
                   {'range': [0, thresholds['low']*100], 'color': "lightgreen"},
                   {'range': [thresholds['low']*100, thresholds['medium']*100], 'color': "orange"},
                   {'range': [thresholds['medium']*100, 100], 'color': "red"}],
               'threshold': {'value': risk_score*100}}
    ))
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Chronic Absenteeism Prediction System")
    
    # Navigation
    page = st.sidebar.selectbox("Choose Page", ["Batch Prediction", "Individual Check", "System Training", "Settings"])
    
    if page == "Batch Prediction":
        batch_prediction()
    elif page == "Individual Check":
        individual_prediction()
    elif page == "System Training":
        system_training()
    elif page == "Settings":
        system_settings()

def system_training():
    """Model training interface"""
    st.header("🔧 Model Training")
    
    uploaded_file = st.file_uploader("Upload training data", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                model, report, shap_data = train_model(df)
                if model:
                    st.session_state.model = model
                    st.success("Model trained successfully!")
                    st.json(report)

def system_settings():
    """System configuration"""
    st.header("⚙️ System Settings")
    
    # Risk thresholds
    st.subheader("Risk Thresholds")
    low = st.slider("Low Risk Threshold", 0.0, 1.0, st.session_state.risk_thresholds['low'])
    medium = st.slider("Medium Risk Threshold", 0.0, 1.0, st.session_state.risk_thresholds['medium'])
    if low >= medium:
        st.error("Low threshold must be less than medium threshold")
    else:
        st.session_state.risk_thresholds = {'low': low, 'medium': medium, 'high': 1.0}
    
    # Interventions
    st.subheader("Intervention Settings")
    for name, details in st.session_state.interventions.items():
        col1, col2 = st.columns(2)
        with col1:
            new_cost = st.number_input(f"{name} Cost", value=details['cost'])
        with col2:
            new_eff = st.slider(f"{name} Effectiveness", 0.0, 1.0, details['effectiveness'])
        st.session_state.interventions[name] = {'cost': new_cost, 'effectiveness': new_eff}

if __name__ == "__main__":
    main()
