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
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Helper functions
def preprocess_data(df, is_training=True):
    """Preprocess the input data for training or prediction"""
    df = df.copy()
    
    # Calculate attendance percentage if not present
    if 'Attendance_Percentage' not in df.columns:
        if 'Present_Days' in df.columns and 'Absent_Days' in df.columns:
            df['Attendance_Percentage'] = (df['Present_Days'] / 
                                         (df['Present_Days'] + df['Absent_Days'])) * 100
    
    # Encode categorical variables
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                if col in st.session_state.label_encoders:
                    # Handle unknown categories
                    df[col] = df[col].apply(lambda x: x if x in st.session_state.label_encoders[col].classes_ else 'Unknown')
                    df[col] = st.session_state.label_encoders[col].transform(df[col])
    
    return df

def train_model(df):
    """Train ensemble model on the provided data"""
    try:
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Convert CA_Status to binary (0/1)
        if df_processed['CA_Status'].dtype == 'object':
            df_processed['CA_Status'] = df_processed['CA_Status'].map({'NO_CA': 0, 'CA': 1}).astype(int)
        elif df_processed['CA_Status'].dtype == 'bool':
            df_processed['CA_Status'] = df_processed['CA_Status'].astype(int)
        
        # Verify binary target
        unique_values = df_processed['CA_Status'].unique()
        if set(unique_values) != {0, 1}:
            st.error(f"Target variable must be binary (0/1). Found values: {unique_values}")
            return None, None, None
        
        # Split data
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train Random Forest model
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
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate SHAP values
        explainer = shap.TreeExplainer(model.named_estimators_['xgb'])
        shap_values = explainer.shap_values(X_train)
        
        return model, report, (explainer, shap_values, X_train)
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None

def predict_ca_risk(input_data, model):
    """Predict CA risk for input data"""
    try:
        if isinstance(input_data, dict):
            # Single student prediction
            df = pd.DataFrame([input_data])
        else:
            # Batch prediction
            df = input_data.copy()
        
        # Preprocess
        df_processed = preprocess_data(df, is_training=False)
        
        # Ensure columns match training data
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            df_processed = df_processed[model.feature_names_in_]
        
        # Predict
        if isinstance(model, (XGBClassifier, VotingClassifier)):
            risk = model.predict_proba(df_processed)[:, 1]
        else:
            risk = model.predict(df_processed)
        
        return risk
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

#%% 1. Updated SHAP Feature Importance Plot (Replace existing plot_shap_summary function)
def plot_shap_summary(explainer, shap_values, features):
    """Create SHAP feature importance plot using Plotly for better readability"""
    st.subheader("SHAP Feature Importance")
    
    # Get feature importance values
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': np.abs(shap_values).mean(0)
    }).sort_values('Importance', ascending=True)
    
    # Create interactive bar plot
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Impact on Chronic Absenteeism Risk',
        labels={'Importance': 'Average Absolute SHAP Value'},
        height=600  # Increased height for better visibility
    )
    
    # Update layout for better readability
    fig.update_layout(
        margin=dict(l=150, r=50, t=60, b=50),  # Adjust left margin for long feature names
        xaxis_title="Average Impact on Model Output",
        yaxis_title="Features",
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)

#%% 2. Adjusted Single Student Analysis Plot Sizing (Update SHAP explanation section)
# Inside Single Student Check section after risk calculation:
                # SHAP explanation with adjusted sizing
                if hasattr(st.session_state.model, 'named_estimators_'):
                    try:
                        xgb_model = st.session_state.model.named_estimators_['xgb']
                        explainer = shap.TreeExplainer(xgb_model)
                        
                        df_processed = preprocess_data(pd.DataFrame([input_data]), is_training=False)
                        
                        if hasattr(xgb_model, 'feature_names_in_'):
                            missing_cols = set(xgb_model.feature_names_in_) - set(df_processed.columns)
                            for col in missing_cols:
                                df_processed[col] = 0
                            df_processed = df_processed[xgb_model.feature_names_in_]
                        
                        shap_values = explainer.shap_values(df_processed)
                        
                        st.subheader("Risk Factor Breakdown")
                        plt.figure(figsize=(10, 4), dpi=100)  # Set explicit figure size and DPI
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            df_processed.iloc[0],
                            matplotlib=True,
                            show=False
                        )
                        st.pyplot(plt.gcf(), use_container_width=True)  # Use container width
                        plt.clf()

#%% 3. Fixed Label Encoding for Unknown Values (Update preprocess_data function)
def preprocess_data(df, is_training=True):
    """Preprocess the input data with handling for unknown categories"""
    df = df.copy()
    
    # Create attendance percentage if missing
    if 'Attendance_Percentage' not in df.columns:
        if 'Present_Days' in df.columns and 'Absent_Days' in df.columns:
            df['Attendance_Percentage'] = (df['Present_Days'] / 
                                         (df['Present_Days'] + df['Absent_Days'])) * 100
    
    # Handle categorical features
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                # Fit new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                # Transform with existing encoder, handling unknown values
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    # Replace unknown categories with most frequent category
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
    
    return df


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
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available for this student")

def generate_geographic_map(df):
    """Generate geographic visualization of risk"""
    if 'Address' not in df.columns:
        st.warning("Address data not available for geographic mapping")
        return
    
    st.subheader("Geographic Risk Distribution")
    
    # Sample geocoding - in production you'd want to cache these results
    geolocator = Nominatim(user_agent="ca_predictor")
    
    # Take a sample to avoid geocoding too many addresses
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
        # Create map centered on first location
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
        st.warning("Could not geocode addresses")

def what_if_analysis(student_data, changes):
    """Perform what-if analysis based on proposed changes"""
    modified_data = student_data.copy()
    for feature, value in changes.items():
        if feature in modified_data:
            modified_data[feature] = value
    
    original_risk = predict_ca_risk(student_data, st.session_state.model)[0]
    new_risk = predict_ca_risk(modified_data, st.session_state.model)[0]
    
    return original_risk, new_risk

def intervention_cost_benefit(students_df):
    """Analyze cost vs benefit of interventions"""
    st.subheader("Intervention Cost-Benefit Analysis")
    
    # Calculate potential reductions
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
    
    # Show table
    st.dataframe(results_df.sort_values('Cost per Case Prevented'))
    
    # Show scatter plot
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

# Title and description
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

# System Training Section
if app_mode == "System Training":
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
            # Read file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = {'Grade', 'Gender', 'Present_Days', 'Absent_Days', 
                           'Meal_Code', 'Academic_Performance', 'CA_Status'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Store historical data for time-series analysis
                if 'Date' in df.columns and 'Student_ID' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.historical_data = df
                    
                    # Build student history dictionary
                    for student_id, group in df.groupby('Student_ID'):
                        st.session_state.student_history[student_id] = group.sort_values('Date')
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Train model
                if st.button("Train Prediction Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes"):
                        model, report, shap_data = train_model(df)
                        
                        if model is not None:
                            st.session_state.model = model
                            st.success("Model trained successfully!")
                            
                            # Show performance metrics
                            st.subheader("Model Performance")
                            st.json({
                                "Accuracy": report['accuracy'],
                                "Precision (CA)": report['1']['precision'],
                                "Recall (CA)": report['1']['recall'],
                                "F1-Score (CA)": report['1']['f1-score']
                            })
                            
                            # Feature importance
                            st.subheader("Top Predictive Factors")
                            feature_importance = pd.DataFrame({
                                'Feature': model.named_estimators_['xgb'].feature_names_in_,
                                'Importance': model.named_estimators_['xgb'].feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feature_importance.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Most Important Features'
                            )
                            st.plotly_chart(fig)
                            
                            # SHAP summary plot
                            if shap_data:
                                explainer, shap_values, X_train = shap_data
                                plot_shap_summary(explainer, shap_values, X_train)
                            
                            # Download model
                            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                                joblib.dump(model, tmp.name)
                                with open(tmp.name, 'rb') as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="ca_model.pkl">Download Trained Model</a>'
                                    st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Batch Prediction Section
elif app_mode == "Batch Prediction":
    st.header("📊 Batch Prediction")
    st.markdown("Upload current student data to predict CA risks.")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
    else:
        # Citywide mode toggle
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
                # Read file
                if uploaded_file.name.endswith('.xlsx'):
                    current_df = pd.read_excel(uploaded_file)
                else:
                    current_df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = {'Student_ID', 'Grade', 'Gender', 'Present_Days', 
                               'Absent_Days', 'Meal_Code', 'Academic_Performance'}
                missing_cols = required_cols - set(current_df.columns)
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Show data preview
                    st.subheader("Current Data Preview")
                    st.dataframe(current_df.head())
                    
                    # Predict
                    if st.button("Predict CA Risks", type="primary"):
                        with st.spinner("Predicting risks..."):
                            # Calculate attendance percentage
                            current_df['Attendance_Percentage'] = (
                                current_df['Present_Days'] / 
                                (current_df['Present_Days'] + current_df['Absent_Days'])
                            ) * 100
                            
                            # Predict
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
                                
                                # Store for analytics
                                st.session_state.current_df = current_df
                                
                                # Show results
                                st.subheader("Prediction Results")
                                
                                # Risk distribution
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
                                
                                # Risk distribution chart
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
                                
                                # Risk by grade
                                fig2 = px.box(
                                    current_df,
                                    x='Grade',
                                    y='CA_Risk',
                                    title='CA Risk Distribution by Grade',
                                    color='Grade'
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Download results
                                csv = current_df.to_csv(index=False)
                                st.download_button(
                                    "Download Predictions",
                                    csv,
                                    "ca_predictions.csv",
                                    "text/csv"
                                )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")



# Single Student Check Section
elif app_mode == "Single Student Check":
    st.header("👤 Single Student Check")
    st.markdown("Check CA risk for an individual student.")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
    else:
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
        
        # Results display (outside the form)
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
                
                # SHAP explanation
                if hasattr(st.session_state.model, 'named_estimators_'):
                    try:
                        xgb_model = st.session_state.model.named_estimators_['xgb']
                        explainer = shap.TreeExplainer(xgb_model)
                        
                        df_processed = preprocess_data(pd.DataFrame([input_data]), is_training=False)
                        
                        if hasattr(xgb_model, 'feature_names_in_'):
                            missing_cols = set(xgb_model.feature_names_in_) - set(df_processed.columns)
                            for col in missing_cols:
                                df_processed[col] = 0
                            df_processed = df_processed[xgb_model.feature_names_in_]
                        
                        shap_values = explainer.shap_values(df_processed)
                        
                        st.subheader("Risk Factor Breakdown")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            df_processed.iloc[0],
                            matplotlib=True,
                            show=False
                        )
                        st.pyplot(fig, bbox_inches='tight')
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {str(e)}")
                
                # Historical trends
                if student_id and student_id in st.session_state.student_history:
                    st.subheader("Historical Trends")
                    plot_student_history(student_id)
                
                # What-if analysis section (outside form)
                st.subheader("What-If Analysis")
                st.markdown("See how changes might affect this student's risk:")
                
                # Initialize what-if parameters
                if 'what_if_params' not in st.session_state:
                    st.session_state.what_if_params = {
                        'attendance': present_days,
                        'performance': academic_performance
                    }
                
                what_if_cols = st.columns(2)
                with what_if_cols[0]:
                    st.session_state.what_if_params['attendance'] = st.slider(
                        "Change attendance days",
                        min_value=0,
                        max_value=present_days + absent_days,
                        value=st.session_state.what_if_params['attendance'],
                        key="what_if_attendance"
                    )
                with what_if_cols[1]:
                    st.session_state.what_if_params['performance'] = st.slider(
                        "Change academic performance",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.what_if_params['performance'],
                        key="what_if_performance"
                    )
                
                # Scenario analysis button
                if st.button("Run Scenario Analysis", key="scenario_button"):
                    changes = {
                        'Present_Days': st.session_state.what_if_params['attendance'],
                        'Absent_Days': (present_days + absent_days) - st.session_state.what_if_params['attendance'],
                        'Academic_Performance': st.session_state.what_if_params['performance']
                    }
                    original_risk, new_risk = what_if_analysis(input_data, changes)
                    
                    st.markdown(f"""
                    ### Scenario Results:
                    - **Original Risk**: {original_risk:.1%}
                    - **New Risk**: {new_risk:.1%}
                    - **Change**: {(new_risk - original_risk):+.1%} points
                    """)
                
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

# Advanced Analytics Section
elif app_mode == "Advanced Analytics":
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
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_avg['Month'],
                    y=monthly_avg['CA_Status']*100,
                    name='CA Rate',
                    yaxis='y2',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title=f'Cohort {selected_year} Monthly Trends',
                    yaxis=dict(title='Attendance Percentage'),
                    yaxis2=dict(
                        title='CA Rate Percentage',
                        overlaying='y',
                        side='right'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No historical data available for cohort analysis")
        
        elif viz_option == "Geographic Risk Mapping":
            generate_geographic_map(df)
        
        elif viz_option == "Intervention Cost-Benefit":
            intervention_cost_benefit(df)

# System Settings Section
elif app_mode == "System Settings":
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
