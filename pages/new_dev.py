import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Chronic Absenteeism Predictor", layout="wide")

# --- Session State Initialization ---
def initialize_state():
    defaults = {
        "model": None,
        "label_encoders": {},
        "X_train_columns": [],
        "risk_thresholds": {"low": 0.3, "medium": 0.7, "high": 1.0},
        "current_df": pd.DataFrame(),
        "model_trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_state()

# --- Helper Functions ---
def preprocess_data(df, is_training=True):
    df = df.copy()
    # Attendance %
    if {'Present_Days', 'Absent_Days'}.issubset(df.columns):
        total_days = df['Present_Days'] + df['Absent_Days']
        df['Attendance_Percentage'] = (df['Present_Days'] / total_days.replace(0, 1)) * 100

    # Encode categorical columns
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
                    df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
    return df

def train_model(df):
    try:
        df = preprocess_data(df, is_training=True)
        if 'CA_Status' not in df.columns:
            st.error("Missing CA_Status column in training data.")
            return None, None
        X = df.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df['CA_Status'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train_columns = X_train.columns.tolist()
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        rf = RandomForestClassifier(n_estimators=50, max_depth=3)
        model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.model_trained = True
        joblib.dump(model, "ca_model.pkl")
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        st.success("Model trained successfully!")
        st.json(report)
        return model, report
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None

def load_model():
    try:
        model = joblib.load("ca_model.pkl")
        st.session_state.model = model
        st.session_state.model_trained = True
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def predict_ca_risk(input_data, model):
    try:
        if isinstance(input_data, (dict, pd.Series)):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df = preprocess_data(df, is_training=False)
        missing_cols = set(st.session_state.X_train_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[st.session_state.X_train_columns]
        probabilities = model.predict_proba(df)[:, 1]
        return probabilities[0] if len(probabilities) == 1 else probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def display_prediction(data):
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

# --- Pages ---
def page_training():
    st.header("🔧 Model Training")
    uploaded_file = st.file_uploader("Upload training data (CSV/XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if st.button("Train Model"):
            train_model(df)

def page_batch():
    st.header("📊 Batch Prediction")
    if not st.session_state.model_trained:
        st.warning("Train a model first in the Model Training page.")
        if st.button("Load Existing Model"):
            load_model()
        return
    uploaded_file = st.file_uploader("Upload student data (CSV/XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if st.button("Run Predictions"):
            df['CA_Risk'] = predict_ca_risk(df, st.session_state.model)
            thresholds = st.session_state.risk_thresholds
            df['Risk_Level'] = pd.cut(
                df['CA_Risk'],
                bins=[0, thresholds['low'], thresholds['medium'], 1],
                labels=['Low', 'Medium', 'High']
            )
            st.session_state.current_df = df
            st.success(f"Predictions complete for {len(df)} students!")
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

def page_individual():
    st.header("👤 Individual Student Check")
    if not st.session_state.model_trained:
        st.warning("Train or load a model first.")
        if st.button("Load Existing Model"):
            load_model()
        return
    # Batch selection
    if not st.session_state.current_df.empty:
        selected_student = st.selectbox("Choose from batch results", 
                                      ["New Student"] + st.session_state.current_df['Student_ID'].astype(str).tolist())
        if selected_student != "New Student":
            student_data = st.session_state.current_df.query("Student_ID == @selected_student").iloc[0]
            display_prediction(student_data.to_dict())
            return
    # Manual input
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

def page_settings():
    st.header("⚙️ System Settings")
    low = st.slider("Low Risk Threshold", 0.0, 1.0, st.session_state.risk_thresholds['low'])
    medium = st.slider("Medium Risk Threshold", 0.0, 1.0, st.session_state.risk_thresholds['medium'])
    if low >= medium:
        st.error("Low threshold must be less than medium threshold")
    else:
        st.session_state.risk_thresholds = {'low': low, 'medium': medium, 'high': 1.0}

# --- Main App ---
def main():
    st.title("Chronic Absenteeism Prediction System")
    page = st.sidebar.radio("Choose Page", ["Model Training", "Batch Prediction", "Individual Check", "Settings"])
    if page == "Model Training":
        page_training()
    elif page == "Batch Prediction":
        page_batch()
    elif page == "Individual Check":
        page_individual()
    elif page == "Settings":
        page_settings()

if __name__ == "__main__":
    main()
