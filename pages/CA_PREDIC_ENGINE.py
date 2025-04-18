# Chronic Absenteeism Predictor - Complete Implementation
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

# Initialize all session state variables
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

# Helper functions
def generate_sample_data():
    """Generate sample data for demonstration purposes"""
    np.random.seed(42)
    num_students = 500
    schools = ['School A', 'School B', 'School C', 'School D']
    grades = range(1, 13)
    meal_codes = ['Free', 'Reduced', 'Paid']
    
    current_df = pd.DataFrame({
        'Student_ID': [f'STD{1000+i}' for i in range(num_students)],
        'School': np.random.choice(schools, num_students),
        'Grade': np.random.choice(grades, num_students),
        'Gender': np.random.choice(['Male', 'Female'], num_students),
        'Present_Days': np.random.randint(80, 180, num_students),
        'Absent_Days': np.random.randint(0, 30, num_students),
        'Meal_Code': np.random.choice(meal_codes, num_students),
        'Academic_Performance': np.random.randint(50, 100, num_students),
        'Address': np.random.choice([
            "100 Main St, Anytown, USA",
            "200 Oak Ave, Somewhere, USA",
            "300 Pine Rd, Nowhere, USA"
        ], num_students)
    })
    
    historical_data = pd.DataFrame()
    for year in [2021, 2022, 2023]:
        year_data = current_df.copy()
        year_data['Date'] = pd.to_datetime(f'{year}-09-01') + pd.to_timedelta(
            np.random.randint(0, 180, num_students), unit='d')
        year_data['Present_Days'] = np.random.randint(80, 180, num_students)
        year_data['Absent_Days'] = np.random.randint(0, 30, num_students)
        year_data['CA_Status'] = np.random.choice([0, 1], num_students, p=[0.8, 0.2])
        historical_data = pd.concat([historical_data, year_data])
    
    return current_df, historical_data

def preprocess_data(df, is_training=True):
    """Preprocess the input data for training or prediction with proper unknown handling"""
    df = df.copy()
    
    if 'Attendance_Percentage' not in df.columns:
        if 'Present_Days' in df.columns and 'Absent_Days' in df.columns:
            df['Attendance_Percentage'] = (df['Present_Days'] / 
                                         (df['Present_Days'] + df['Absent_Days'])) * 100
    
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                if col == 'Gender':
                    classes = list(df[col].unique()) + ['Unknown']
                    le.fit(classes)
                else:
                    le.fit(df[col])
                df[col] = le.transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                if col in st.session_state.label_encoders:
                    df[col] = df[col].apply(lambda x: x if x in st.session_state.label_encoders[col].classes_ else 'Unknown')
                    df[col] = st.session_state.label_encoders[col].transform(df[col])
    
    return df

def train_model(df):
    """Train ensemble model on the provided data"""
    try:
        df_processed = preprocess_data(df)
        
        if df_processed['CA_Status'].dtype == 'object':
            df_processed['CA_Status'] = df_processed['CA_Status'].map({'NO_CA': 0, 'CA': 1}).astype(int)
        elif df_processed['CA_Status'].dtype == 'bool':
            df_processed['CA_Status'] = df_processed['CA_Status'].astype(int)
        
        unique_values = df_processed['CA_Status'].unique()
        if set(unique_values) != {0, 1}:
            st.error(f"Target variable must be binary (0/1). Found values: {unique_values}")
            return None, None, None
        
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
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
        
        model = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf)],
            voting='soft'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        explainer = shap.TreeExplainer(model.named_estimators_['xgb'])
        shap_values = explainer.shap_values(X_train)
        
        return model, report, (explainer, shap_values, X_train)
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None

def predict_ca_risk(input_data, model):
    """Predict CA risk for input data with proper error handling"""
    try:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        df_processed = preprocess_data(df, is_training=False)
        
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            df_processed = df_processed[model.feature_names_in_]
        
        if isinstance(model, (XGBClassifier, VotingClassifier)):
            risk = model.predict_proba(df_processed)[:, 1]
        else:
            risk = model.predict(df_processed)
        
        return risk
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def plot_feature_importance(model):
    """Create interactive feature importance visualization"""
    try:
        if hasattr(model, 'named_estimators_'):
            xgb_model = model.named_estimators_['xgb']
            importance = xgb_model.feature_importances_
            features = xgb_model.feature_names_in_
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
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
            
            fig.update_layout(
                margin=dict(l=100, r=50, t=80, b=50),
                xaxis_title="Importance Score",
                yaxis_title="Features",
                yaxis={'categoryorder':'total ascending'},
                hovermode="y"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate feature importance plot: {str(e)}")

def plot_student_history(student_id):
    """Plot historical trends for a student"""
    if student_id in st.session_state.student_history:
        history = st.session_state.student_history[student_id]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history['Date'],
            y=history['Attendance_Percentage'],
            name='Attendance %',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=history['Date'],
            y=history['CA_Risk']*100,
            name='CA Risk %',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Student {student_id} Historical Trends',
            yaxis=dict(title='Attendance Percentage'),
            yaxis2=dict(
                title='CA Risk Percentage',
                overlaying='y',
                side='right'
            ) )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available for this student")

def generate_geographic_map(df):
    """Generate geographic visualization of risk"""
    if 'Address' not in df.columns:
        st.warning("Address data not available for geographic mapping")
        return
    
    st.subheader("Geographic Risk Distribution")
    
    if df['Address'].isnull().all():
        sample_df, _ = generate_sample_data()
        df['Address'] = sample_df['Address'].sample(len(df)).values
    
    geolocator = Nominatim(user_agent="ca_predictor")
    sample_df = df.sample(min(50, len(df)))
    
    locations = []
    for idx, row in sample_df.iterrows():
        try:
            location = geolocator.geocode(row['Address'])
            if location:
                locations.append({
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'risk': row['CA_Risk'],
                    'student': row.get('Student_ID', '')
                })
        except:
            continue
    
    if locations:
        m = folium.Map(location=[locations[0]['lat'], locations[0]['lon']], zoom_start=12)
        
        for loc in locations:
            color = '#ff4b4b' if loc['risk'] > 0.7 else '#ffa500' if loc['risk'] > 0.3 else '#2ecc71'
            folium.CircleMarker(
                location=[loc['lat'], loc['lon']],
                radius=5 + (loc['risk'] * 10),
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Student: {loc['student']}<br>Risk: {loc['risk']:.2f}"
            ).add_to(m)
        
        folium_static(m)
    else:
        st.warning("Could not geocode addresses. Using sample locations.")
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
        folium.Marker(
            location=[37.7749, -122.4194],
            popup="Sample Location 1"
        ).add_to(m)
        folium_static(m)

def what_if_analysis(student_data):
    """Perform what-if analysis without page refresh"""
    st.subheader("What-If Analysis")
    
    if 'what_if_params' not in st.session_state:
        st.session_state.what_if_params = {
            'present': student_data.get('Present_Days', 90),
            'absent': student_data.get('Absent_Days', 10),
            'performance': student_data.get('Academic_Performance', 75)
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.what_if_params['present'] = st.slider(
            "Present Days", 
            min_value=0, 
            max_value=200,
            value=st.session_state.what_if_params['present'],
            key="wi_present"
        )
        
    with col2:
        st.session_state.what_if_params['absent'] = st.slider(
            "Absent Days",
            min_value=0,
            max_value=200,
            value=st.session_state.what_if_params['absent'],
            key="wi_absent"
        )
    
    st.session_state.what_if_params['performance'] = st.slider(
        "Academic Performance",
        min_value=0,
        max_value=100,
        value=st.session_state.what_if_params['performance'],
        key="wi_performance"
    )
    
    if st.button("Calculate New Risk", key="wi_calculate"):
        modified_data = student_data.copy()
        modified_data['Present_Days'] = st.session_state.what_if_params['present']
        modified_data['Absent_Days'] = st.session_state.what_if_params['absent']
        modified_data['Academic_Performance'] = st.session_state.what_if_params['performance']
        
        original_risk = predict_ca_risk(student_data, st.session_state.model)[0]
        new_risk = predict_ca_risk(modified_data, st.session_state.model)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Risk", f"{original_risk:.1%}")
        with col2:
            st.metric("New Risk", f"{new_risk:.1%}", 
                     delta=f"{(new_risk - original_risk):+.1%}")
        
        st.session_state.what_if_changes = {
            'original': original_risk,
            'new': new_risk,
            'change': new_risk - original_risk
        }

def intervention_cost_benefit(students_df):
    """Analyze cost vs benefit of interventions"""
    st.subheader("Intervention Cost-Benefit Analysis")
    
    interventions = st.session_state.interventions
    thresholds = st.session_state.risk_thresholds
    
    high_risk = students_df[students_df['CA_Risk'] >= thresholds['medium']]
    num_high_risk = len(high_risk)
    
    results = []
    for name, details in interventions.items():
        cost_per_student = details['cost']
        effectiveness = details['effectiveness']
        
        total_cost = cost_per_student * num_high_risk
        potential_reduction = num_high_risk * effectiveness
        cost_per_reduction = total_cost / potential_reduction if potential_reduction > 0 else float('inf')
        
        results.append({
            'Intervention': name,
            'Total Cost': total_cost,
            'Potential Cases Prevented': round(potential_reduction),
            'Cost per Case Prevented': round(cost_per_reduction),
            'Effectiveness': effectiveness
        })
    
    results_df = pd.DataFrame(results)
    
    st.dataframe(results_df.sort_values('Cost per Case Prevented'))
    
    fig = px.scatter(
        results_df,
        x='Potential Cases Prevented',
        y='Total Cost',
        size='Effectiveness',
        color='Intervention',
        hover_name='Intervention',
        title='Intervention Cost vs Effectiveness'
    )
    st.plotly_chart(fig, use_container_width=True)

# Application Sections
def system_training():
    """System Training section"""
    st.header("🔧 System Training")
    st.markdown("Upload historical data to train the prediction model.")
    
    with st.expander("📋 Data Requirements", expanded=True):
        st.markdown("""
        Your Excel file should include these columns:
        - **Student_ID**: Unique identifier
        - **School**: School name/code
        - **Grade**: Grade level (1-12)
        - **Gender**: Male/Female/Other
        - **Present_Days**: Number of days present
        - **Absent_Days**: Number of days absent
        - **Meal_Code**: Free/Reduced/Paid (SES proxy)
        - **Academic_Performance**: Score (0-100)
        - **CA_Status**: Chronic Absenteeism status (YES/NO or 1/0)
        - **Date**: (Optional) For time-series analysis
        - **Address**: (Optional) For geographic mapping
        """)
    
    uploaded_file = st.file_uploader(
        "Upload Historical Data (Excel)", 
        type=["xlsx", "csv"],
        help="Upload 2+ years of historical attendance data"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            required_cols = {'Grade', 'Gender', 'Present_Days', 'Absent_Days', 
                           'Meal_Code', 'Academic_Performance', 'CA_Status'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if 'Date' in df.columns and 'Student_ID' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.historical_data = df
                    
                    for student_id, group in df.groupby('Student_ID'):
                        st.session_state.student_history[student_id] = group.sort_values('Date')
                
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                if st.button("Train Prediction Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes"):
                        model, report, shap_data = train_model(df)
                        
                        if model is not None:
                            st.session_state.model = model
                            st.success("Model trained successfully!")
                            
                            st.subheader("Model Performance")
                            st.json({
                                "Accuracy": report['accuracy'],
                                "Precision (CA)": report['1']['precision'],
                                "Recall (CA)": report['1']['recall'],
                                "F1-Score (CA)": report['1']['f1-score']
                            })
                            
                            plot_feature_importance(model)
                            
                            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                                joblib.dump(model, tmp.name)
                                with open(tmp.name, 'rb') as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="ca_model.pkl">Download Trained Model</a>'
                                    st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def batch_prediction():
    """Batch Prediction section"""
    st.header("📊 Batch Prediction")
    st.markdown("Upload current student data to predict CA risks.")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
    else:
        st.session_state.citywide_mode = st.checkbox(
            "Enable Citywide Mode (track students across schools)",
            value=st.session_state.citywide_mode
        )
        
        uploaded_file = st.file_uploader(
            "Upload Current Student Data (Excel)", 
            type=["xlsx", "csv"],
            help="Upload current term data for prediction"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    current_df = pd.read_excel(uploaded_file)
                else:
                    current_df = pd.read_csv(uploaded_file)
                
                required_cols = {'Student_ID', 'Grade', 'Gender', 'Present_Days', 
                               'Absent_Days', 'Meal_Code', 'Academic_Performance'}
                missing_cols = required_cols - set(current_df.columns)
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.subheader("Current Data Preview")
                    st.dataframe(current_df.head())
                    
                    if st.button("Predict CA Risks", type="primary"):
                        with st.spinner("Predicting risks..."):
                            current_df['Attendance_Percentage'] = (
                                current_df['Present_Days'] / 
                                (current_df['Present_Days'] + current_df['Absent_Days'])
                            ) * 100
                            
                            risks = predict_ca_risk(current_df, st.session_state.model)
                            
                            if risks is not None:
                                current_df['CA_Risk'] = risks
                                current_df['CA_Risk_Level'] = pd.cut(
                                    current_df['CA_Risk'],
                                    bins=[0, st.session_state.risk_thresholds['low'], 
                                          st.session_state.risk_thresholds['medium'], 
                                          st.session_state.risk_thresholds['high']],
                                    labels=['Low', 'Medium', 'High']
                                )
                                
                                st.session_state.current_df = current_df
                                
                                st.subheader("Prediction Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    low_count = (current_df['CA_Risk_Level'] == 'Low').sum()
                                    st.metric("Low Risk", low_count)
                                with col2:
                                    medium_count = (current_df['CA_Risk_Level'] == 'Medium').sum()
                                    st.metric("Medium Risk", medium_count)
                                with col3:
                                    high_count = (current_df['CA_Risk_Level'] == 'High').sum()
                                    st.metric("High Risk", high_count)
                                
                                fig1 = px.pie(
                                    current_df,
                                    names='CA_Risk_Level',
                                    title='Risk Level Distribution',
                                    color='CA_Risk_Level',
                                    color_discrete_map={
                                        'Low': '#2ecc71',
                                        'Medium': '#f39c12',
                                        'High': '#e74c3c'
                                    }
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                                
                                fig2 = px.box(
                                    current_df,
                                    x='Grade',
                                    y='CA_Risk',
                                    title='CA Risk Distribution by Grade',
                                    color='Grade'
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                csv = current_df.to_csv(index=False)
                                st.download_button(
                                    "Download Predictions",
                                    csv,
                                    "ca_predictions.csv",
                                    "text/csv"
                                )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def single_student_check():
    """Single Student Check section with all fixes"""
    st.header("👤 Single Student Check")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
        return
    
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
            
            if st.session_state.citywide_mode and transferred and prev_ca == "Yes":
                risk = min(risk * 1.4, 0.99)
            
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
            
            plot_feature_importance(st.session_state.model)
            
            if student_id and student_id in st.session_state.student_history:
                st.subheader("Historical Trends")
                plot_student_history(student_id)
            
            what_if_analysis(input_data)
            
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

def advanced_analytics():
    """Advanced Analytics section"""
    st.header("📈 Advanced Analytics")
    st.markdown("Interactive visualizations for deeper insights.")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
    elif st.session_state.current_df.empty:
        st.warning("No prediction data available. Please run batch predictions first.")
        if st.button("Generate Sample Data for Demo"):
            current_df, historical_data = generate_sample_data()
            st.session_state.current_df = current_df
            st.session_state.historical_data = historical_data
            st.success("Sample data generated! Refresh the page to view analytics.")
    else:
        df = st.session_state.current_df
        
        viz_option = st.selectbox(
            "Select Visualization",
            [
                "Risk Distribution by School",
                "Attendance vs. Academic Performance",
                "Risk Heatmap by Grade & SES",
                "Temporal Attendance Trends",
                "Cohort Analysis",
                "Geographic Risk Mapping",
                "Intervention Cost-Benefit"
            ]
        )
        
        if viz_option == "Risk Distribution by School":
            st.subheader("Risk Distribution by School")
            
            if 'School' not in df.columns:
                st.warning("School data not available in predictions.")
            else:
                school_stats = df.groupby('School').agg({
                    'CA_Risk': 'mean',
                    'Student_ID': 'count'
                }).rename(columns={'Student_ID': 'Student_Count'})
                
                fig = px.scatter(
                    school_stats,
                    x='Student_Count',
                    y='CA_Risk',
                    size='Student_Count',
                    color='CA_Risk',
                    hover_name=school_stats.index,
                    title='CA Risk Distribution by School',
                    labels={
                        'CA_Risk': 'Average CA Risk',
                        'Student_Count': 'Number of Students'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Attendance vs. Academic Performance":
            st.subheader("Attendance vs. Academic Performance")
            
            fig = px.scatter(
                df,
                x='Attendance_Percentage',
                y='Academic_Performance',
                color='CA_Risk_Level',
                hover_name='Student_ID',
                title='Attendance vs. Academic Performance',
                color_discrete_map={
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Risk Heatmap by Grade & SES":
            st.subheader("Risk Heatmap by Grade & SES")
            
            if 'Meal_Code' not in df.columns:
                st.warning("Meal Code (SES) data not available in predictions.")
            else:
                try:
                    if 'Meal_Code' in st.session_state.label_encoders:
                        le = st.session_state.label_encoders['Meal_Code']
                        df['Meal_Code'] = le.inverse_transform(df['Meal_Code'])
                except:
                    pass
                
                heatmap_data = df.pivot_table(
                    values='CA_Risk',
                    index='Grade',
                    columns='Meal_Code',
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Meal Code", y="Grade", color="CA Risk"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    title='Average CA Risk by Grade and Socioeconomic Status',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Temporal Attendance Trends":
            st.subheader("Temporal Attendance Trends")
            
            if not st.session_state.historical_data.empty:
                trend_data = st.session_state.historical_data.groupby('Date').agg({
                    'Attendance_Percentage': 'mean',
                    'CA_Status': 'mean'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_data['Date'],
                    y=trend_data['Attendance_Percentage'],
                    name='Attendance Rate',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=trend_data['Date'],
                    y=trend_data['CA_Status']*100,
                    name='CA Rate',
                    yaxis='y2',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Historical Attendance and CA Trends',
                    yaxis=dict(title='Attendance Percentage'),
                    yaxis2=dict(
                        title='CA Rate Percentage',
                        overlaying='y',
                        side='right'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No historical temporal data available")
        
        elif viz_option == "Cohort Analysis":
            st.subheader("Cohort Analysis")
            
            if not st.session_state.historical_data.empty:
                years = st.session_state.historical_data['Date'].dt.year.unique()
                selected_year = st.selectbox("Select Cohort Year", sorted(years))
                
                cohort = st.session_state.historical_data[
                    st.session_state.historical_data['Date'].dt.year == selected_year
                ]
                
                cohort_students = cohort['Student_ID'].unique()
                cohort_trends = st.session_state.historical_data[
                    st.session_state.historical_data['Student_ID'].isin(cohort_students)
                ]
                
                cohort_trends['Month'] = cohort_trends['Date'].dt.to_period('M')
                monthly_avg = cohort_trends.groupby('Month').agg({
                    'Attendance_Percentage': 'mean',
                    'CA_Status': 'mean'
                }).reset_index()
                monthly_avg['Month'] = monthly_avg['Month'].astype(str)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_avg['Month'],
                    y=monthly_avg['Attendance_Percentage'],
                    name='Attendance Rate',
                    line=dict(color='blue'))
                    
                fig.add_trace(go.Scatter(
                    x=monthly_avg['Month'],
                    y=monthly_avg['CA_Status']*100,
                    name='CA Rate',
                    yaxis='y2',
                    line=dict(color='red'))
                
                fig.update_layout(
                    title=f'Cohort {selected_year} Monthly Trends',
                    yaxis=dict(title='Attendance Percentage'),
                    yaxis2=dict(
                        title='CA Rate Percentage',
                        overlaying='y',
                        side='right'
                    ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No historical data available for cohort analysis")
        
        elif viz_option == "Geographic Risk Mapping":
            generate_geographic_map(df)
        
        elif viz_option == "Intervention Cost-Benefit":
            intervention_cost_benefit(df)

def system_settings():
    """System Settings section"""
    st.header("⚙️ System Settings")
    
    st.subheader("Risk Threshold Configuration")
    st.markdown("Adjust the probability thresholds for risk levels:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        low_thresh = st.slider(
            "Low Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_thresholds['low'],
            step=0.05,
            key="low_thresh"
        )
    with col2:
        medium_thresh = st.slider(
            "Medium Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_thresholds['medium'],
            step=0.05,
            key="medium_thresh"
        )
    with col3:
        high_thresh = st.slider(
            "High Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_thresholds['high'],
            step=0.05,
            disabled=True,
            key="high_thresh"
        )
    
    if low_thresh >= medium_thresh:
        st.error("Low risk threshold must be less than medium threshold")
    else:
        st.session_state.risk_thresholds = {
            'low': low_thresh,
            'medium': medium_thresh,
            'high': high_thresh
        }
        st.success("Thresholds updated successfully!")
    
    st.subheader("Intervention Configuration")
    st.markdown("Configure available interventions and their parameters:")
    
    interventions = st.session_state.interventions
    for name, details in interventions.items():
        st.markdown(f"**{name}**")
        col1, col2 = st.columns(2)
        with col1:
            new_cost = st.number_input(
                f"Cost per student ({name})",
                min_value=0,
                value=details['cost'],
                key=f"cost_{name}"
            )
        with col2:
            new_effect = st.slider(
                f"Effectiveness ({name})",
                min_value=0.0,
                max_value=1.0,
                value=details['effectiveness'],
                step=0.05,
                key=f"eff_{name}"
            )
        interventions[name] = {'cost': new_cost, 'effectiveness': new_effect}
    
    st.session_state.interventions = interventions

# Main application
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
