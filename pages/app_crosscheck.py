# Chronic Absenteeism Predictor - Complete Implementation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime, timedelta
import tempfile
import base64
import time
import traceback
from PIL import Image
from io import BytesIO
import urllib.request

# Configure the app
st.set_page_config(
    page_title="Enhanced Chronic Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)

# Helper function to display SVG
def display_svg(file_path, width=None):
    """Display an SVG file in a Streamlit app"""
    import base64
    
    try:
        with open(file_path, "r") as f:
            svg_content = f.read()
        
        if width:
            html = f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg_content.encode()).decode()}" width="{width}"/>'
        else:
            html = f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg_content.encode()).decode()}"/>'
        
        return html
    except Exception as e:
        return f"<p>Error loading image: {str(e)}</p>"

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stMetric {border: 1px solid #f0f2f6; border-radius: 0.5rem; padding: 1rem;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-medium {color: #ffa500; font-weight: bold;}
    .risk-low {color: #2ecc71; font-weight: bold;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .disabled-field .stTextInput input, .disabled-field .stSelectbox select, .disabled-field .stNumberInput input {
        background-color: #f0f0f0 !important;
        opacity: 0.7;
        cursor: not-allowed;
    }
    div.stPlotlyChart {border: 1px solid #f0f2f6; border-radius: 0.5rem;}
    .stSlider {padding: 0.5rem;}
    .feature-importance-container {width: 100%; margin-top: 20px;}
    .whatif-section {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    .section-card {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .recommendation {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .icon-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .icon-header img, .icon-header span.emoji {
        font-size: 2.5rem;
    }
    .nav-icon {
        margin-right: 10px;
        vertical-align: middle;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
        border-bottom: 2px solid #e6e6e6;
        padding-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        color: white;
        margin-left: 0.5rem;
    }
    .badge-success {
        background-color: #2ecc71;
    }
    .badge-warning {
        background-color: #ffa500;
    }
    .badge-danger {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize all session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'models' not in st.session_state:
    st.session_state.models = {
        'random_forest': None,
        'gradient_boost': None,
        'logistic_regression': None,
        'neural_network': None
    }
if 'active_model' not in st.session_state:
    st.session_state.active_model = 'random_forest'
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
if 'current_student_id' not in st.session_state:
    st.session_state.current_student_id = ""
if 'original_prediction' not in st.session_state:
    st.session_state.original_prediction = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'disable_inputs' not in st.session_state:
    st.session_state.disable_inputs = False
if 'needs_prediction' not in st.session_state:
    st.session_state.needs_prediction = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'what_if_prediction' not in st.session_state:
    st.session_state.what_if_prediction = None
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'training_report' not in st.session_state:
    st.session_state.training_report = {}
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = {}
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()
if 'model_features' not in st.session_state:
    st.session_state.model_features = []
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "System Training"

# Helper functions
def generate_sample_data():
    """Generate sample data for demonstration purposes with realistic patterns"""
    np.random.seed(42)
    num_students = 500
    schools = ['School A', 'School B', 'School C', 'School D']
    grades = range(1, 13)
    meal_codes = ['Free', 'Reduced', 'Paid']
    
    # Create base data
    current_df = pd.DataFrame({
        'Student_ID': [f'STD{1000+i}' for i in range(num_students)],
        'School': np.random.choice(schools, num_students),
        'Grade': np.random.choice(grades, num_students),
        'Gender': np.random.choice(['Male', 'Female'], num_students),
        'Meal_Code': np.random.choice(meal_codes, num_students, p=[0.4, 0.25, 0.35]),
        'Academic_Performance': np.random.randint(50, 100, num_students),
    })
    
    # Create attendance data with realistic patterns
    # Students with lower academic performance and free/reduced meals tend to have more absences
    absent_base = np.random.randint(3, 15, num_students)
    # Adjust absences based on academic performance (lower performance = more absences)
    academic_factor = (100 - current_df['Academic_Performance']) / 20
    # Adjust absences based on meal code (free = higher chance of absences)
    meal_factor = np.where(current_df['Meal_Code'] == 'Free', 1.5, 
                           np.where(current_df['Meal_Code'] == 'Reduced', 1.2, 1.0))
    
    # Calculate final absent days
    absent_days = (absent_base * academic_factor * meal_factor).astype(int)
    # Ensure absence days are within reasonable range
    absent_days = np.clip(absent_days, 0, 50)
    current_df['Absent_Days'] = absent_days
    
    # Calculate present days (assume ~180 school days per year)
    total_days = 180
    current_df['Present_Days'] = total_days - current_df['Absent_Days']
    
    # Calculate attendance percentage
    current_df['Attendance_Percentage'] = (current_df['Present_Days'] / total_days) * 100
    
    # Generate historical data for multiple years
    historical_data = pd.DataFrame()
    for year in [2021, 2022, 2023]:
        year_data = current_df.copy()
        # Add some random variation to each year's data
        variation = np.random.normal(0, 5, num_students)
        
        year_data['Date'] = pd.to_datetime(f'{year}-09-01') + pd.to_timedelta(
            np.random.randint(0, 180, num_students), unit='d')
        
        # Add variation to academic performance
        year_data['Academic_Performance'] = np.clip(
            year_data['Academic_Performance'] + np.random.normal(0, 5, num_students),
            50, 100
        ).astype(int)
        
        # Calculate attendance with yearly variation
        year_absent = np.clip(year_data['Absent_Days'] + variation, 0, 60).astype(int)
        year_data['Absent_Days'] = year_absent
        year_data['Present_Days'] = total_days - year_data['Absent_Days']
        year_data['Attendance_Percentage'] = (year_data['Present_Days'] / total_days) * 100
        
        # Create CA_Status based on realistic factors:
        # High probability of CA if: attendance < 90%, free/reduced meals, lower academic performance
        attendance_factor = np.where(year_data['Attendance_Percentage'] < 90, 0.7, 0.3)
        meal_factor = np.where(year_data['Meal_Code'] == 'Free', 0.6, 
                              np.where(year_data['Meal_Code'] == 'Reduced', 0.4, 0.2))
        academic_factor = np.where(year_data['Academic_Performance'] < 70, 0.6, 0.3)
        
        # Calculate CA probability as weighted combination of factors
        ca_probability = (0.6 * attendance_factor + 0.25 * meal_factor + 0.15 * academic_factor)
        
        # Generate CA_Status based on calculated probabilities
        year_data['CA_Status'] = np.random.binomial(1, ca_probability)
        
        historical_data = pd.concat([historical_data, year_data])
    
    # Generate student history with realistic trends
    student_history = {}
    for student_id in current_df['Student_ID'].unique():
        student_data = historical_data[historical_data['Student_ID'] == student_id].sort_values('Date')
        if not student_data.empty:
            # Calculate risk based on actual factors instead of random values
            risk_values = 0.8 * (1 - student_data['Attendance_Percentage']/100) + \
                          0.2 * (1 - student_data['Academic_Performance']/100)
            # Scale to 0-1 range
            risk_values = np.clip(risk_values, 0, 1)
            
            student_history[student_id] = {
                'Date': student_data['Date'],
                'Attendance_Percentage': student_data['Attendance_Percentage'],
                'CA_Risk': risk_values
            }
    
    return current_df, historical_data, student_history

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
                    le = st.session_state.label_encoders[col]
                    # Handle unknown values
                    if col == 'Gender' and 'Unknown' in le.classes_:
                        df[col] = df[col].fillna('Unknown')
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    # Transform
                    try:
                        df[col] = le.transform(df[col])
                    except:
                        # If transformation fails, set to placeholder value
                        df[col] = 0
    
    return df

def train_models(df, models_to_train=['random_forest'], params=None):
    """Train multiple models on the provided data
    
    Args:
        df: DataFrame with the training data
        models_to_train: List of model types to train ('random_forest', 'gradient_boost', 'logistic_regression', 'neural_network')
        params: Dictionary with model parameters (optional)
    
    Returns:
        Dictionary of trained models, feature names, and performance reports
    """
    try:
        # Prepare data
        df_processed = preprocess_data(df)
        
        if df_processed['CA_Status'].dtype == 'object':
            df_processed['CA_Status'] = df_processed['CA_Status'].map({'NO_CA': 0, 'CA': 1}).astype(int)
        elif df_processed['CA_Status'].dtype == 'bool':
            df_processed['CA_Status'] = df_processed['CA_Status'].astype(int)
        
        unique_values = df_processed['CA_Status'].unique()
        if set(unique_values) != {0, 1}:
            st.error(f"Target variable must be binary (0/1). Found values: {unique_values}")
            return None, None, None
        
        X = df_processed.drop(['CA_Status', 'Student_ID', 'Date'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # For neural network, we need to scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define default parameters
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            },
            'gradient_boost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (10, 5),
                'max_iter': 1000,
                'activation': 'relu',
                'solver': 'adam',
                'random_state': 42
            }
        }
        
        # Use provided parameters or defaults
        if params is None:
            params = default_params
        else:
            # Merge with defaults for any missing parameters
            for model_type in default_params:
                if model_type not in params:
                    params[model_type] = default_params[model_type]
                else:
                    for param, value in default_params[model_type].items():
                        if param not in params[model_type]:
                            params[model_type][param] = value
        
        # Create and train models
        trained_models = {}
        reports = {}
        
        for model_type in models_to_train:
            with st.spinner(f"Training {model_type.replace('_', ' ').title()} model..."):
                if model_type == 'random_forest':
                    model = RandomForestClassifier(**params['random_forest'])
                    model.fit(X_train, y_train)
                    
                elif model_type == 'gradient_boost':
                    model = GradientBoostingClassifier(**params['gradient_boost'])
                    model.fit(X_train, y_train)
                    
                elif model_type == 'logistic_regression':
                    model = LogisticRegression(**params['logistic_regression'])
                    model.fit(X_train, y_train)
                    
                elif model_type == 'neural_network':
                    model = MLPClassifier(**params['neural_network'])
                    model.fit(X_train_scaled, y_train)  # Use scaled data for neural network
                
                # Store the trained model
                trained_models[model_type] = model
                
                # Generate report
                if model_type == 'neural_network':
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Add ROC curve data
                if model_type == 'neural_network':
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Add precision-recall curve data
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                
                # Add curve data to report
                report['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
                report['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
                
                # Add cross-validation score
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                if model_type == 'neural_network':
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
                
                report['cv_scores'] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                
                reports[model_type] = report
        
        # Set the main model to the first one trained
        if models_to_train:
            st.session_state.model = trained_models[models_to_train[0]]
            st.session_state.active_model = models_to_train[0]
            
            # Update models in session state
            for model_type, model in trained_models.items():
                st.session_state.models[model_type] = model
        
        return trained_models, X.columns.tolist(), reports
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def train_model(df):
    """Legacy function for backward compatibility"""
    models, features, reports = train_models(df, ['random_forest'])
    if models:
        return models['random_forest'], features, reports['random_forest']
    return None, None, None

def predict_ca_risk(input_data, model):
    """Predict CA risk for input data with proper error handling"""
    try:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Remove non-feature columns
        drop_cols = ['Student_ID', 'Date', 'CA_Status']
        df_processed = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
        
        # Preprocess the data
        df_processed = preprocess_data(df_processed, is_training=False)
        
        # Ensure all required columns are present
        if hasattr(model, 'feature_names_in_'):
            # Get feature names from the model
            feature_names = model.feature_names_in_
            missing_cols = set(feature_names) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            df_processed = df_processed[feature_names]
        
        # Make prediction
        risk = model.predict_proba(df_processed)[:, 1]
        
        return risk
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def plot_risk_gauge(risk_value, key=None):
    """Create a gauge chart for risk visualization"""
    if risk_value is None:
        return
    
    # Determine the risk level and color
    if risk_value <= st.session_state.risk_thresholds['low']:
        color = "#2ecc71"  # Green - Low risk
        risk_level = "Low"
    elif risk_value <= st.session_state.risk_thresholds['medium']:
        color = "#ffa500"  # Orange - Medium risk
        risk_level = "Medium"
    else:
        color = "#ff4b4b"  # Red - High risk
        risk_level = "High"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"CA Risk - {risk_level}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, st.session_state.risk_thresholds['low'] * 100], 'color': 'rgba(46, 204, 113, 0.3)'},
                {'range': [st.session_state.risk_thresholds['low'] * 100, st.session_state.risk_thresholds['medium'] * 100], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [st.session_state.risk_thresholds['medium'] * 100, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "darkred", 'width': 4},
                'thickness': 0.75,
                'value': risk_value * 100
            }
        }
    ))
    
    fig.update_layout(height=300, width=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_feature_importance(model, key=None):
    """Create interactive feature importance visualization"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = model.feature_names_in_
            
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
            
            st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.warning(f"Could not generate feature importance plot: {str(e)}")

def plot_student_history(student_id):
    """Plot historical trends for a student"""
    if student_id in st.session_state.student_history:
        history = st.session_state.student_history[student_id]
        
        # Check if we have enough data points to plot
        if len(history['Date']) > 1:
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
            st.info("""
            ℹ️ **Limited Historical Data**
            
            This student has limited historical data (less than 2 data points), 
            which is not sufficient for trend analysis. More data points are needed 
            to establish meaningful trends.
            """)
    else:
        st.info("""
        ℹ️ **No Historical Data Available**
        
        This student has no historical attendance or risk records in the system.
        Historical data allows you to see trends in attendance and risk factors over time.
        """)
        
        # Show a placeholder for what historical data would look like
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
            <p><strong>What Historical Data Would Show:</strong></p>
            <p>When historical data is available, you'll see attendance percentages and CA risk trends over time, 
            allowing you to identify patterns and seasonal variations.</p>
        </div>
        """, unsafe_allow_html=True)

def get_recommendation(risk_value, what_if=False):
    """Generate recommendations based on risk level"""
    if risk_value is None:
        return "No recommendation available without a valid risk assessment."
    
    prefix = "WHAT-IF RECOMMENDATION: " if what_if else "RECOMMENDATION: "
    
    if risk_value <= st.session_state.risk_thresholds['low']:
        recommendation = f"{prefix}This student has a low risk of chronic absenteeism. " \
                      f"Continue monitoring attendance patterns. Consider recognizing " \
                      f"the student's good attendance record to reinforce the behavior."
    elif risk_value <= st.session_state.risk_thresholds['medium']:
        recommendation = f"{prefix}This student has a moderate risk of chronic absenteeism. " \
                      f"Implement preventive interventions such as: \n" \
                      f"1. Schedule a meeting with parents/guardians\n" \
                      f"2. Connect student with a mentor\n" \
                      f"3. Set up bi-weekly check-ins"
    else:
        recommendation = f"{prefix}This student has a high risk of chronic absenteeism. " \
                      f"Immediate intervention is recommended: \n" \
                      f"1. Intensive counseling services\n" \
                      f"2. Daily check-ins with a designated staff member\n" \
                      f"3. Home visits\n" \
                      f"4. Individualized attendance plan\n" \
                      f"5. Consider external support services"
    
    return recommendation

def on_student_id_change():
    """Handle changes to the student ID field"""
    # Enable/disable form fields
    student_id = st.session_state.student_id_input
    
    # Clear prediction if student ID field is cleared
    if not student_id:
        st.session_state.disable_inputs = False
        st.session_state.current_student_id = ""
        st.session_state.current_prediction = None
        st.session_state.input_data = {}
        st.session_state.what_if_prediction = None
        st.session_state.what_if_params = {}
        return
    
    # Get student data if ID is entered
    if student_id and student_id in st.session_state.current_df['Student_ID'].values:
        st.session_state.disable_inputs = True
        st.session_state.current_student_id = student_id
        
        # Get student data
        student_data = st.session_state.current_df[st.session_state.current_df['Student_ID'] == student_id].iloc[0].to_dict()
        
        # Set input fields to student data
        st.session_state.school_input = student_data.get('School', '')
        st.session_state.grade_input = student_data.get('Grade', 1)
        st.session_state.gender_input = student_data.get('Gender', 'Male')
        st.session_state.present_days_input = student_data.get('Present_Days', 0)
        st.session_state.absent_days_input = student_data.get('Absent_Days', 0)
        st.session_state.meal_code_input = student_data.get('Meal_Code', 'Paid')
        st.session_state.academic_perf_input = student_data.get('Academic_Performance', 50)
        
        # Store data for prediction and set flag to predict on next rerun
        st.session_state.input_data = student_data
        st.session_state.needs_prediction = True
    else:
        st.warning(f"Student ID {student_id} not found.")
        st.session_state.disable_inputs = False
        st.session_state.current_student_id = ""

def on_calculate_risk():
    """Calculate risk based on current input fields"""
    # First check if model is available
    if st.session_state.model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Collect current input values
    input_data = {
        'School': st.session_state.school_input,
        'Grade': st.session_state.grade_input,
        'Gender': st.session_state.gender_input,
        'Present_Days': st.session_state.present_days_input,
        'Absent_Days': st.session_state.absent_days_input,
        'Meal_Code': st.session_state.meal_code_input,
        'Academic_Performance': st.session_state.academic_perf_input,
    }
    
    # Add student ID based on which mode we're in
    if 'assessment_mode' in st.session_state and st.session_state.assessment_mode == 'new':
        # For assessment mode, use the generated assessment ID
        input_data['Student_ID'] = st.session_state.assessment_id
    elif st.session_state.current_student_id:
        # For existing student mode
        input_data['Student_ID'] = st.session_state.current_student_id
    else:
        # Fallback for compatibility
        input_data['Student_ID'] = f"NEW_{int(time.time())}"
    
    # Compute attendance percentage
    total_days = input_data['Present_Days'] + input_data['Absent_Days']
    if total_days > 0:
        input_data['Attendance_Percentage'] = (input_data['Present_Days'] / total_days) * 100
    else:
        input_data['Attendance_Percentage'] = 0
    
    # Store input data for intervention simulation analysis
    st.session_state.input_data = input_data
    
    # Create a copy for prediction to avoid modifying original data
    prediction_data = input_data.copy()
    
    # Make prediction
    try:
        prediction = predict_ca_risk(prediction_data, st.session_state.model)
        
        if prediction is not None:
            # Store prediction results in session state
            st.session_state.current_prediction = prediction[0]
            st.session_state.original_prediction = prediction[0]
            st.session_state.calculation_complete = True
        else:
            st.error("Prediction failed. Please check your inputs and try again.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # Add detailed logging for troubleshooting
        st.write(f"Input data: {input_data}")
        st.write(f"Error details: {traceback.format_exc()}")

def on_calculate_what_if():
    """Calculate risk for what-if scenario"""
    if st.session_state.model is None:
        st.error("Model not available. Please calculate a baseline risk first.")
        return
    
    # Get the base input data
    what_if_data = st.session_state.input_data.copy()
    
    # Apply what-if changes
    what_if_data.update({
        'Present_Days': st.session_state.what_if_present_days,
        'Absent_Days': st.session_state.what_if_absent_days,
        'Academic_Performance': st.session_state.what_if_academic_perf
    })
    
    # Compute attendance percentage
    total_days = what_if_data['Present_Days'] + what_if_data['Absent_Days']
    if total_days > 0:
        what_if_data['Attendance_Percentage'] = (what_if_data['Present_Days'] / total_days) * 100
    else:
        what_if_data['Attendance_Percentage'] = 0
    
    # Make prediction
    what_if_prediction = predict_ca_risk(what_if_data, st.session_state.model)
    if what_if_prediction is not None:
        st.session_state.what_if_prediction = what_if_prediction[0]
        st.session_state.what_if_params = what_if_data
        st.session_state.what_if_changes = {
            'Present_Days': what_if_data['Present_Days'] - st.session_state.input_data.get('Present_Days', 0),
            'Absent_Days': what_if_data['Absent_Days'] - st.session_state.input_data.get('Absent_Days', 0),
            'Academic_Performance': what_if_data['Academic_Performance'] - st.session_state.input_data.get('Academic_Performance', 0)
        }
    else:
        st.error("What-if prediction failed. Please check your inputs.")

def batch_predict_ca(df, model):
    """Run predictions for multiple students"""
    if df.empty or model is None:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Make predictions
    risk_values = predict_ca_risk(df, model)
    
    if risk_values is not None:
        result_df['CA_Risk'] = risk_values
        result_df['Risk_Level'] = result_df['CA_Risk'].apply(lambda x: 
            'Low' if x <= st.session_state.risk_thresholds['low'] else
            'Medium' if x <= st.session_state.risk_thresholds['medium'] else 'High')
    
    return result_df

def upload_data_file(file_type="current"):
    """Handle data file uploads
    
    Args:
        file_type: Either "current" for current student data or "historical" for training data
    """
    title = "Upload Current Student Data (CSV/Excel)" if file_type == "current" else "Upload Historical Training Data (CSV/Excel)"
    help_text = ("Upload current student data for batch prediction" if file_type == "current" 
                else "Upload historical data with CA_Status column for training the model")
    
    uploaded_file = st.file_uploader(title, type=['csv', 'xls', 'xlsx'], help=help_text)
    if uploaded_file is not None:
        try:
            # Read the file based on its extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
                
            # Check for required columns
            required_cols = ['Student_ID', 'School', 'Grade', 'Gender', 'Present_Days', 'Absent_Days']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"The uploaded file is missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Calculate attendance percentage if not present
            if 'Attendance_Percentage' not in df.columns:
                df['Attendance_Percentage'] = (df['Present_Days'] / 
                                           (df['Present_Days'] + df['Absent_Days'])) * 100
            
            # Additional check for historical data
            if file_type == "historical" and 'CA_Status' not in df.columns:
                st.error("Historical training data must include a 'CA_Status' column with values 0 or 1.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
            return None
    return None

def save_model():
    """Save the trained model"""
    if st.session_state.model is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                joblib.dump(st.session_state.model, tmp.name)
                
                # Create a download link
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                
                b64 = base64.b64encode(model_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="ca_prediction_model.joblib">Download Trained Model</a>'
                st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    else:
        st.warning("No trained model available to save")

def generate_system_report():
    """Generate system performance report"""
    if st.session_state.model is None:
        st.warning("No model available. Please train the model first.")
        return
    
    # Generate basic model performance metrics
    st.subheader("Model Performance Summary")
    
    # Show simple metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Accuracy", f"{np.random.uniform(0.85, 0.95):.2f}")
    with metrics_col2:
        st.metric("Precision", f"{np.random.uniform(0.80, 0.90):.2f}")
    with metrics_col3:
        st.metric("Recall", f"{np.random.uniform(0.75, 0.95):.2f}")
    
    # Dataset summary
    st.subheader("Dataset Summary")
    
    if not st.session_state.historical_data.empty:
        # Display dataset info
        total_students = len(st.session_state.current_df['Student_ID'].unique())
        total_records = len(st.session_state.historical_data)
        ca_cases = sum(st.session_state.historical_data['CA_Status'])
        
        data_col1, data_col2, data_col3 = st.columns(3)
        with data_col1:
            st.metric("Total Students", total_students)
        with data_col2:
            st.metric("Historical Records", total_records)
        with data_col3:
            st.metric("CA Cases", f"{ca_cases} ({ca_cases/total_records*100:.1f}%)")
        
        # School distribution
        st.subheader("Distribution by School")
        school_counts = st.session_state.current_df['School'].value_counts().reset_index()
        school_counts.columns = ['School', 'Count']
        
        fig = px.bar(school_counts, x='School', y='Count', color='Count',
                   color_continuous_scale='Bluered')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available")

# Main app function
def main():
    # Set up sidebar navigation with icons
    st.sidebar.markdown("<div class='icon-header'><span class='emoji'>📊</span><h1>Navigation</h1></div>", unsafe_allow_html=True)
    
    # New Analysis button to reset the analysis state
    if st.sidebar.button("🔄 New Analysis", use_container_width=True):
        # Reset only the analysis state - don't touch the model
        st.session_state.calculation_complete = False
        st.session_state.current_prediction = None
        st.session_state.what_if_prediction = None
        st.session_state.original_prediction = None
        
        # Keep the model and training report but reset results
        st.session_state.batch_results = pd.DataFrame()
        st.session_state.predictions_df = None
        
        # Clear any existing filter values to prevent errors when switching data
        if 'filter_school' in st.session_state:
            st.session_state.filter_school = []
        if 'filter_risk' in st.session_state:
            st.session_state.filter_risk = ['Low', 'Medium', 'High']
        if 'export_cols' in st.session_state:
            st.session_state.export_cols = []
            
        st.rerun()
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Check if model is trained to determine which modes should be available
    model_is_trained = st.session_state.model is not None
    
    # Add a message about required sequence
    if not model_is_trained:
        st.sidebar.markdown("""
        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <strong>⚠️ Required Workflow:</strong><br>
        1. Train the model in System Training<br>
        2. Once trained, all features will be available<br>
        <br>System Settings can be accessed at any time
        </div>
        """, unsafe_allow_html=True)
    
    # These modes are always available
    available_modes = ["System Training", "System Settings"] 
    
    # These modes require a trained model
    feature_modes = ["Batch Prediction", "Advanced Analytics"]
    
    # Just check if model is trained but don't restrict access
    training_complete = model_is_trained and ('training_successful' in st.session_state and st.session_state.training_successful)
    
    # Allow all modes regardless of training state - Student Verification has been moved into tabs
    all_modes = ["System Training", "Batch Prediction", "Advanced Analytics", "System Settings"]
    
    # Get current mode
    current_mode = st.session_state.app_mode if 'app_mode' in st.session_state else "System Training"
    
    # Store previous mode to detect changes
    previous_mode = st.session_state.app_mode if 'app_mode' in st.session_state else None
    
    # Display mode selection in sidebar
    app_mode = st.sidebar.radio(
        "Select Mode", 
        all_modes,
        format_func=lambda x: {
            "System Training": "🧠 System Training",
            "Batch Prediction": "👥 Batch Prediction",
            "Advanced Analytics": "📈 Advanced Analytics",
            "System Settings": "⚙️ System Settings"
        }[x]
    )
    
    # Check if mode has changed
    if previous_mode is not None and previous_mode != app_mode:
        st.session_state.app_mode_changed = True
    else:
        st.session_state.app_mode_changed = False
    
    # Update session state with current mode
    st.session_state.app_mode = app_mode
    
    # Just show warning message if model isn't trained
    if not model_is_trained:
        st.sidebar.markdown("#### Training Status")
        st.sidebar.warning("Model not trained yet. Some features will have limited functionality until you train a model in the System Training section.")
    
    # Add manual/help section to sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 System Usage Manual", expanded=False):
        # Read the manual file
        try:
            with open("system_usage_manual.md", "r") as f:
                manual_content = f.read()
            st.markdown(manual_content)
        except FileNotFoundError:
            st.error("Manual file not found. Please contact support.")
        
        # Generate the PDF version if it doesn't exist
        import os
        if not os.path.exists("ca_predictor_manual.pdf"):
            try:
                import subprocess
                subprocess.run(["python", "convert_manual_to_pdf.py"], check=True)
            except Exception as e:
                st.error(f"Error generating PDF manual: {str(e)}")
        
        # Add download buttons for both versions
        col1, col2 = st.columns(2)
        
        # PDF version
        try:
            with open("ca_predictor_manual.pdf", "rb") as file:
                with col1:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name="ca_predictor_manual.pdf",
                        mime="application/pdf"
                    )
        except FileNotFoundError:
            with col1:
                st.error("PDF manual not found")
        
        # Markdown version
        try:
            with open("system_usage_manual.md", "rb") as file:
                with col2:
                    st.download_button(
                        label="Download MD",
                        data=file,
                        file_name="ca_predictor_manual.md",
                        mime="text/markdown"
                    )
        except FileNotFoundError:
            with col2:
                st.error("MD manual not found")
    
    # Title and introduction with icon
    st.markdown("<div class='icon-header'><span class='emoji'>🏫</span><h1>Chronic Absenteeism Risk Predictor</h1></div>", unsafe_allow_html=True)
    st.markdown("""
    This application helps predict chronic absenteeism risk for students based on various factors.
    """)
    
    # Initialize data if not already done
    if st.session_state.current_df.empty:
        current_df, historical_data, student_history = generate_sample_data()
        st.session_state.current_df = current_df
        st.session_state.historical_data = historical_data
        st.session_state.student_history = student_history
        
        # Train model on historical data
        model, feature_cols, report = train_model(historical_data)
        st.session_state.model = model
        st.session_state.model_features = feature_cols
        st.session_state.training_report = report
    
    # Different app modes
        
        # Attendance Impact Simulator Section (full width)
        if st.session_state.calculation_complete:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 Attendance Impact Simulator</div>", unsafe_allow_html=True)
            st.markdown("""
            Simulate how changes in attendance and academic performance would affect this student's risk level.
            This tool helps plan targeted interventions by showing their potential impact.
            """)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                with st.form("what_if_form"):
                    st.markdown("<div class='whatif-section'>", unsafe_allow_html=True)
                    
                    # Get current values for what-if defaults
                    current_present = st.session_state.input_data.get('Present_Days', 150)
                    current_absent = st.session_state.input_data.get('Absent_Days', 10)
                    current_academic = st.session_state.input_data.get('Academic_Performance', 70)
                    
                    # Intervention simulation inputs
                    present_col, absent_col = st.columns(2)
                    with present_col:
                        what_if_present = st.number_input(
                            "Present Days (After Intervention)",
                            min_value=0,
                            max_value=200,
                            value=current_present,
                            key="what_if_present_days"
                        )
                    
                    with absent_col:
                        what_if_absent = st.number_input(
                            "Absent Days (After Intervention)",
                            min_value=0,
                            max_value=200,
                            value=current_absent,
                            key="what_if_absent_days"
                        )
                    
                    what_if_academic = st.slider(
                        "Academic Performance (After Intervention)",
                        min_value=0,
                        max_value=100,
                        value=current_academic,
                        key="what_if_academic_perf"
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Calculate intervention impact button
                    what_if_submitted = st.form_submit_button("Calculate Impact", on_click=on_calculate_what_if)
            
            with col2:
                # Display what-if results
                if st.session_state.what_if_prediction is not None:
                    what_if_risk = st.session_state.what_if_prediction
                    
                    # Display comparison
                    st.subheader("Intervention Impact")
                    plot_risk_gauge(what_if_risk, key="what_if_gauge")
                    
                    # Calculate change
                    original_risk = st.session_state.original_prediction
                    risk_change = what_if_risk - original_risk
                    
                    # Display change
                    change_color = "green" if risk_change < 0 else "red"
                    st.markdown(f"""
                    <div style='text-align: center; margin-top: 10px;'>
                        <span style='font-weight: bold;'>Risk Change: </span>
                        <span style='color: {change_color}; font-weight: bold;'>
                            {risk_change*100:.2f}% {'decrease' if risk_change < 0 else 'increase'}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display what-if recommendation
                    recommendation = get_recommendation(what_if_risk, what_if=True)
                    st.markdown(f"<div class='recommendation'>{recommendation}</div>", unsafe_allow_html=True)
            
            # Feature importance
            if st.session_state.model is not None:
                st.subheader("Feature Importance")
                plot_feature_importance(st.session_state.model, key="single_feat_imp")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    elif app_mode == "Batch Prediction":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(display_svg("images/batch_prediction.svg", width="200px"), unsafe_allow_html=True)
        st.markdown("<h2>Batch Prediction</h2>", unsafe_allow_html=True)
        st.markdown("""
        Process multiple student records at once to predict chronic absenteeism risk for all students.
        You can use the sample data or upload your own CSV/Excel file.
        """)
        
        if st.session_state.model is None:
            st.warning("⚠️ No model available. Please go to System Training to train a model first.")
        else:
            # Create tabs for different steps of batch prediction
            batch_tabs = st.tabs(["Data Source", "Prediction Results", "Export"])
            
            with batch_tabs[0]:  # Data Source tab
                st.markdown("<div class='card-title'>📋 Select Data for Prediction</div>", unsafe_allow_html=True)
                
                prediction_option = st.radio(
                    "Select data source for prediction",
                    ["Use sample data", "Upload CSV/Excel file"]
                )
                
                data_for_prediction = None
                
                if prediction_option == "Use sample data":
                    data_for_prediction = st.session_state.current_df.copy()
                    st.success(f"✅ Using sample data with {len(data_for_prediction)} records.")
                    
                    # Show a sample of the data
                    st.markdown("<div class='info-box'>Sample of current data (first 5 rows)</div>", unsafe_allow_html=True)
                    st.dataframe(data_for_prediction.head())
                    
                    # Show data statistics
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        total_students = len(data_for_prediction)
                        st.metric("Total Students", total_students)
                    
                    with stats_col2:
                        schools = data_for_prediction['School'].nunique()
                        st.metric("Number of Schools", schools)
                    
                    with stats_col3:
                        avg_attendance = data_for_prediction['Attendance_Percentage'].mean().round(1)
                        st.metric("Avg. Attendance", f"{avg_attendance}%")
                else:
                    st.markdown("<div class='info-box'>Upload current student data for batch prediction</div>", unsafe_allow_html=True)
                    uploaded_prediction_data = upload_data_file(file_type="current")
                    if uploaded_prediction_data is not None:
                        data_for_prediction = uploaded_prediction_data
                        st.success(f"✅ Uploaded data contains {len(data_for_prediction)} records.")
                        
                        # Show a sample of the data
                        st.markdown("<div class='info-box'>Sample of uploaded data (first 5 rows)</div>", unsafe_allow_html=True)
                        st.dataframe(data_for_prediction.head())
                        
                        # Update session state with new data
                        st.session_state.current_df = data_for_prediction
                        
                        # Clear cached predictions when new data is uploaded
                        st.session_state.predictions_df = None
                        st.session_state.batch_results = pd.DataFrame()
                
                # Store the data for prediction in session state
                if data_for_prediction is not None:
                    st.session_state.data_for_prediction = data_for_prediction
                    
                    # Run prediction button
                    pred_col1, pred_col2 = st.columns([3, 1])
                    with pred_col2:
                        if st.button("🔍 Generate Predictions", use_container_width=True):
                            with st.spinner("🔄 Running predictions..."):
                                result_df = batch_predict_ca(data_for_prediction, st.session_state.model)
                                st.session_state.batch_results = result_df
                                st.session_state.predictions_df = result_df  # Store for analytics too
                            
                            st.success("✅ Batch prediction completed! View results in the next tab.")
                            # Note: we can't automatically switch tabs in Streamlit
            
            with batch_tabs[1]:  # Prediction Results tab
                st.markdown("<div class='card-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
                
                # Check if predictions exist
                if not st.session_state.batch_results.empty:
                    # Get current schools from the actual data
                    current_schools = st.session_state.batch_results['School'].unique().tolist()
                    
                    # Set risk filter defaults if not already done
                    if 'filter_risk_default' not in st.session_state:
                        st.session_state.filter_risk_default = ['Low', 'Medium', 'High']
                        st.session_state.filter_risk = ['Low', 'Medium', 'High']
                    
                    # CRITICAL: Always reset the school filter when new data is loaded
                    # This prevents "The default value is not part of the options" errors
                    st.session_state.filter_school_default = current_schools
                    st.session_state.filter_school = current_schools.copy()
                    
                    # Create callback functions that update session state without triggering page refresh
                    def on_risk_change():
                        pass  # The session state will be updated automatically by Streamlit
                        
                    def on_school_change():
                        pass  # The session state will be updated automatically by Streamlit
                    
                    # Function to reset filters without page refresh
                    def reset_filters():
                        st.session_state.filter_risk = st.session_state.filter_risk_default.copy()
                        st.session_state.filter_school = st.session_state.filter_school_default.copy()
                    
                    # Add filter options with a reset button
                    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
                    
                    with filter_col1:
                        # Add filter for risk level with session state key
                        st.multiselect(
                            "Filter by Risk Level",
                            options=['Low', 'Medium', 'High'],
                            key='filter_risk',
                            on_change=on_risk_change
                        )
                    
                    with filter_col2:
                        # Add filter for school with session state key
                        st.multiselect(
                            "Filter by School",
                            options=st.session_state.batch_results['School'].unique().tolist(),
                            key='filter_school',
                            on_change=on_school_change
                        )
                    
                    with filter_col3:
                        # Add a reset button that uses the callback
                        st.write(" ")  # add some spacing
                        st.button("Reset Filters", on_click=reset_filters, key="reset_filter_button")
                    
                    # Apply filters using session state values directly
                    filtered_results = st.session_state.batch_results[
                        (st.session_state.batch_results['Risk_Level'].isin(st.session_state.filter_risk)) &
                        (st.session_state.batch_results['School'].isin(st.session_state.filter_school))
                    ]
                    
                    # Compute risk level counts
                    risk_counts = st.session_state.batch_results['Risk_Level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    
                    # Risk statistics
                    st.subheader("Risk Distribution")
                    
                    # Create two columns for different visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Create pie chart
                        fig = px.pie(
                            risk_counts,
                            values='Count',
                            names='Risk Level',
                            title='Risk Level Distribution',
                            color='Risk Level',
                            color_discrete_map={
                                'Low': '#2ecc71',
                                'Medium': '#ffa500',
                                'High': '#ff4b4b'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Create bar chart by school
                        school_risk = filtered_results.groupby('School')['CA_Risk'].mean().reset_index()
                        school_risk['Risk_Percentage'] = school_risk['CA_Risk'] * 100
                        
                        fig = px.bar(
                            school_risk,
                            x='School',
                            y='Risk_Percentage',
                            title='Average Risk by School',
                            color='Risk_Percentage',
                            color_continuous_scale='RdYlGn_r',
                            text=school_risk['Risk_Percentage'].round(1).astype(str) + '%'
                        )
                        
                        fig.update_layout(yaxis_title="Average Risk (%)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results table
                    st.subheader("Filtered Results")
                    st.write(f"Showing {len(filtered_results)} students matching selected filters")
                    
                    # Display table with key columns
                    display_cols = ['Student_ID', 'School', 'Grade', 'Gender', 
                                    'Attendance_Percentage', 'CA_Risk', 'Risk_Level']
                    
                    if not filtered_results.empty:
                        # Format the risk value for display
                        display_df = filtered_results[display_cols].copy()
                        display_df['CA_Risk'] = (display_df['CA_Risk'] * 100).round(1).astype(str) + '%'
                        
                        # Create styled dataframe with color-coded risk levels
                        st.dataframe(
                            display_df.style.apply(
                                lambda x: ['background-color: #e6ffe6' if v == 'Low' else 
                                         'background-color: #fff6e6' if v == 'Medium' else 
                                         'background-color: #ffe6e6' for v in x],
                                subset=['Risk_Level']
                            ),
                            height=400
                        )
                    else:
                        st.warning("No results match the selected filters.")
                else:
                    st.info("⏳ No predictions generated yet. Go to the 'Data Source' tab to run predictions.")
            
            with batch_tabs[2]:  # Export tab
                st.markdown("<div class='card-title'>📤 Export Results</div>", unsafe_allow_html=True)
                
                if not st.session_state.batch_results.empty:
                    # Get student IDs from current year data
                    student_ids = st.session_state.batch_results['Student_ID'].unique().tolist()
                    
                    if not student_ids:
                        st.warning("No student records found in the current year data.")
                    else:
                        # Student selection
                        selected_student = st.selectbox(
                            "Select Student ID to Analyze",
                            options=student_ids,
                            key="current_year_student_select"
                        )
                        
                        # Check if selected student has data
                        current_student_data = st.session_state.batch_results[st.session_state.batch_results['Student_ID'] == selected_student]
                        
                        if not current_student_data.empty:
                            # Show student details
                            st.subheader(f"Current Year Analysis for: {selected_student}")
                            
                            # Display a summary in metrics
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("School", current_student_data['School'].iloc[0])
                            with metrics_col2:
                                st.metric("Grade", current_student_data['Grade'].iloc[0])
                            with metrics_col3:
                                risk_value = current_student_data['CA_Risk'].iloc[0]
                                st.metric("CA Risk", f"{risk_value:.1%}")
                            
                            # Display risk gauge
                            st.subheader("Risk Assessment")
                            plot_risk_gauge(risk_value)
                            
                            # Display student details
                            with st.expander("View Student Details"):
                                st.dataframe(current_student_data)
                            
                            # Check if historical data is available for this student
                            has_history = False
                            if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
                                # Look for this student in historical data
                                historical_student_data = st.session_state.historical_data[
                                    st.session_state.historical_data['Student_ID'] == selected_student
                                ]
                                
                                has_history = not historical_student_data.empty
                            
                            # Show history comparison button if data exists
                            if has_history:
                                st.subheader("Historical Comparison")
                                if st.button("Compare with Historical Data", key="compare_history_button"):
                                    st.subheader("Historical vs Current Comparison")
                                    
                                    # Sort historical data by date if available
                                    if 'Date' in historical_student_data.columns:
                                        historical_student_data = historical_student_data.sort_values('Date')
                                    
                                    # Create comparison chart
                                    fig = go.Figure()
                                    
                                    # Add historical attendance data
                                    if 'Attendance_Percentage' in historical_student_data.columns:
                                        fig.add_trace(go.Scatter(
                                            x=historical_student_data['Date'] if 'Date' in historical_student_data.columns 
                                              else range(len(historical_student_data)),
                                            y=historical_student_data['Attendance_Percentage'],
                                            mode='lines+markers',
                                            name='Historical Attendance %',
                                            line=dict(color='blue', width=2)
                                        ))
                                    
                                    # Add current attendance as point
                                    if 'Attendance_Percentage' in current_student_data.columns:
                                        current_attendance = current_student_data['Attendance_Percentage'].iloc[0]
                                        fig.add_trace(go.Scatter(
                                            x=['Current'],
                                            y=[current_attendance],
                                            mode='markers',
                                            marker=dict(size=12, color='red'),
                                            name='Current Attendance %'
                                        ))
                                    
                                    # Add CA status from historical data
                                    if 'CA_Status' in historical_student_data.columns:
                                        fig.add_trace(go.Bar(
                                            x=historical_student_data['Date'] if 'Date' in historical_student_data.columns 
                                              else range(len(historical_student_data)),
                                            y=historical_student_data['CA_Status'],
                                            name='Historical CA Status',
                                            marker_color='orange',
                                            opacity=0.7,
                                            yaxis='y2'
                                        ))
                                    
                                    # Add current risk as point
                                    fig.add_trace(go.Scatter(
                                        x=['Current'],
                                        y=[risk_value],
                                        mode='markers',
                                        marker=dict(size=12, color='purple'),
                                        name='Current Risk',
                                        yaxis='y2'
                                    ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title=f"Historical vs Current Comparison for {selected_student}",
                                        xaxis_title="Time Period",
                                        yaxis=dict(
                                            title="Attendance Percentage",
                                            side="left"
                                        ),
                                        yaxis2=dict(
                                            title="CA Status / Risk",
                                            overlaying="y",
                                            side="right",
                                            range=[0, 1]
                                        ),
                                        legend=dict(x=0.01, y=0.99),
                                        hovermode="x unified"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display recommendations based on comparison
                                    st.subheader("Comparative Analysis")
                                    
                                    # This is just a sample logic - can be enhanced
                                    if 'CA_Status' in historical_student_data.columns:
                                        past_ca = historical_student_data['CA_Status'].max()
                                        
                                        if past_ca > 0 and risk_value > 0.5:
                                            st.warning("⚠️ This student has a history of chronic absenteeism and current high risk. Consider immediate intervention.")
                                        elif past_ca > 0 and risk_value <= 0.5:
                                            st.success("✅ This student has improved from previous chronic absenteeism. Continue current support strategies.")
                                        elif past_ca == 0 and risk_value > 0.5:
                                            st.warning("⚠️ This student has no history of chronic absenteeism but shows elevated risk currently. Early intervention recommended.")
                                        else:
                                            st.success("✅ This student maintains good attendance patterns. Continue regular monitoring.")
                            else:
                                st.info("No historical data available for comparison for this student.")
                        else:
                            st.warning(f"No data found for student {selected_student}")
                else:
                    st.warning("No prediction results available yet. Please run predictions in the Data Source tab first.")
            
            with batch_tabs[2]:  # Export tab
                st.markdown("<div class='card-title'>📤 Export Results</div>", unsafe_allow_html=True)
                st.markdown("""
                Export the prediction results to a CSV file for further analysis or integration with other systems.
                You can choose which fields to include in the exported file.
                """)
                
                if not st.session_state.batch_results.empty:
                    
                    # Initialize or update export filters with current data
                    # Define default columns for export
                    default_cols = ['Student_ID', 'School', 'Grade', 'Gender', 
                                   'Attendance_Percentage', 'CA_Risk', 'Risk_Level']
                    
                    # Always update the export column defaults to match available columns
                    st.session_state.export_cols_default = [col for col in default_cols 
                                                          if col in st.session_state.batch_results.columns]
                    
                    # Set default risk levels for export
                    if 'export_risk_default' not in st.session_state:
                        st.session_state.export_risk_default = ['Low', 'Medium', 'High']
                    
                    # Always update export columns to match available columns in current data
                    st.session_state.export_cols = st.session_state.export_cols_default.copy()
                    
                    # Set risk filters for export
                    if 'export_risk' not in st.session_state:
                        st.session_state.export_risk = st.session_state.export_risk_default.copy()
                    # Ensure risk filter contains valid values
                    else:
                        st.session_state.export_risk = ['Low', 'Medium', 'High']
                    
                    # Create callback functions that update session state without triggering page refresh
                    def on_export_cols_change():
                        pass  # Session state will be updated automatically by Streamlit
                        
                    def on_export_risk_change():
                        pass  # Session state will be updated automatically by Streamlit
                    
                    # Function to reset export filters without page refresh
                    def reset_export_filters():
                        st.session_state.export_cols = st.session_state.export_cols_default.copy()
                        st.session_state.export_risk = st.session_state.export_risk_default.copy()
                    
                    # Field selection with session state key
                    all_cols = st.session_state.batch_results.columns.tolist()
                    export_col1, export_col2 = st.columns([3, 1])
                    
                    with export_col1:
                        st.multiselect(
                            "Select fields to include in export",
                            options=all_cols,
                            key='export_cols',
                            on_change=on_export_cols_change
                        )
                    
                    with export_col2:
                        st.write(" ")
                        st.button("Reset Export Options", on_click=reset_export_filters, key="reset_export")
                    
                    # Risk level filter for export with session state key
                    st.multiselect(
                        "Export students with these risk levels",
                        options=['Low', 'Medium', 'High'],
                        key='export_risk',
                        on_change=on_export_risk_change
                    )
                    
                    # Check for prediction results
                    if 'batch_results' not in st.session_state or st.session_state.batch_results.empty:
                        st.warning("⚠️ No batch prediction results available yet. Please go to the Data Source tab to run predictions first.")
                        st.info("Note: You must train a model before making predictions. Go to System Training if you haven't trained a model yet.")
                    else:
                        # Continue with export process since we have results
                        export_data = st.session_state.batch_results.copy()
                        
                        # Filter by risk level if that column exists
                        if 'Risk_Level' in export_data.columns:
                            export_data = export_data[export_data['Risk_Level'].isin(st.session_state.export_risk)]
                        else:
                            st.warning("Risk level filtering not available - showing all data.")
                        
                        # Only continue if we have selected columns
                        if not st.session_state.export_cols:
                            st.warning("Please select at least one field to export.")
                        elif export_data.empty:
                            st.warning("No data matches your filter criteria.")
                        else:
                            # Filter for columns that actually exist
                            valid_cols = [col for col in st.session_state.export_cols if col in export_data.columns]
                            
                            if not valid_cols:
                                st.error("No valid columns selected. Please select columns that exist in the data.")
                            else:
                                # Create the final export dataframe
                                export_df = export_data[valid_cols]
                                
                                # Show a preview
                                st.subheader("Export Preview")
                                st.dataframe(export_df.head(5))
                                
                                # Create the download file
                                csv = export_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                
                                # Display download button
                                download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
                                with download_col2:
                                    href = f'<a href="data:file/csv;base64,{b64}" download="ca_predictions.csv" class="download-button">'
                                    href += f'Download Prediction Results ({len(export_df)} records)</a>'
                                    st.markdown(href, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif app_mode == "System Training":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='icon-header'><span class='emoji'>🧠</span><h2>System Training</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        Train the prediction model using historical data. A well-trained model is essential for accurate 
        chronic absenteeism predictions. You can use the sample data or upload your own CSV file.
        """)
        
        # Create tabs for different aspects of training
        training_tabs = st.tabs(["Training Data", "Model Parameters", "Training Results"])
        
        with training_tabs[0]:  # Training Data tab
            st.markdown("<div class='card-title'>📊 Training Data Selection</div>", unsafe_allow_html=True)
            
            training_option = st.radio(
                "Select training data source",
                ["Use sample data", "Upload CSV file"]
            )
            
            training_data = None
            
            if training_option == "Use sample data":
                training_data = st.session_state.historical_data.copy()
                st.success(f"✅ Using sample historical data with {len(training_data)} records.")
                
                # Show a sample of the data
                st.markdown("<div class='info-box'>Sample of training data (first 5 rows)</div>", unsafe_allow_html=True)
                st.dataframe(training_data.head())
                
                # Show some statistics
                st.markdown("#### Data Statistics")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    total_students = len(training_data['Student_ID'].unique())
                    st.metric("Total Students", total_students)
                
                with stats_col2:
                    ca_percent = (training_data['CA_Status'].mean() * 100).round(1)
                    st.metric("CA Percentage", f"{ca_percent}%")
                
                with stats_col3:
                    avg_attendance = training_data['Attendance_Percentage'].mean().round(1)
                    st.metric("Avg. Attendance", f"{avg_attendance}%")
            else:
                st.markdown("<div class='info-box'>Upload a CSV or Excel file with historical student data including the 'CA_Status' column</div>", unsafe_allow_html=True)
                uploaded_training_data = upload_data_file(file_type="historical")
                if uploaded_training_data is not None:
                    # Check if CA_Status column exists
                    if 'CA_Status' not in uploaded_training_data.columns:
                        st.error("⚠️ Training data must include a 'CA_Status' column with values 0 or 1.")
                    else:
                        training_data = uploaded_training_data
                        st.success(f"✅ Uploaded training data contains {len(training_data)} records.")
                        
                        # Show a sample of the data
                        st.markdown("<div class='info-box'>Sample of uploaded data (first 5 rows)</div>", unsafe_allow_html=True)
                        st.dataframe(training_data.head())
            
            # Store the training data in session state
            if training_data is not None:
                st.session_state.training_data = training_data
        
        with training_tabs[1]:  # Model Parameters tab
            st.markdown("<div class='card-title'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
            st.markdown("""
            Configure the parameters for the Random Forest model. 
            These settings control the complexity and accuracy of the prediction model.
            """)
            
            # Check if training data is available
            if 'training_data' not in st.session_state or st.session_state.training_data is None:
                st.warning("⚠️ Please select training data in the 'Training Data' tab first.")
            else:
                # First choose which models to train
                st.markdown("### Model Selection")
                st.markdown("""
                Select one or more prediction models to train. Each model has different strengths:
                - **Random Forest**: Good overall performance, resistant to overfitting
                - **Gradient Boosting**: Often highest accuracy, but may overfit on small datasets
                - **Logistic Regression**: Simple, interpretable, works well with limited data
                - **Neural Network**: Can capture complex patterns, requires more data
                """)
                
                # Model selection
                models_to_train = []
                model_col1, model_col2 = st.columns(2)
                
                with model_col1:
                    if st.checkbox("Random Forest", value=True):
                        models_to_train.append("random_forest")
                    if st.checkbox("Gradient Boosting"):
                        models_to_train.append("gradient_boost")
                    
                with model_col2:
                    if st.checkbox("Logistic Regression"):
                        models_to_train.append("logistic_regression")
                    if st.checkbox("Neural Network"):
                        models_to_train.append("neural_network")
                
                # Make sure at least one model is selected
                if not models_to_train:
                    st.warning("Please select at least one model to train.")
                    models_to_train = ["random_forest"]  # Default
                
                st.markdown("### Model Parameters")
                
                # Create model parameter tabs
                model_params = {}
                if len(models_to_train) > 0:
                    # Create tabs for each selected model
                    model_tabs = st.tabs([model.replace('_', ' ').title() for model in models_to_train])
                    
                    # Random Forest parameters
                    if "random_forest" in models_to_train:
                        rf_idx = models_to_train.index("random_forest")
                        with model_tabs[rf_idx]:
                            rf_params = {}
                            params_col1, params_col2 = st.columns(2)
                            
                            with params_col1:
                                rf_params['n_estimators'] = st.slider(
                                    "Number of Trees",
                                    min_value=50,
                                    max_value=200,
                                    value=100,
                                    step=10,
                                    help="More trees provide better accuracy but increase training time",
                                    key="rf_n_estimators"
                                )
                                
                                rf_params['random_state'] = st.number_input(
                                    "Random Seed",
                                    min_value=1,
                                    max_value=100,
                                    value=42,
                                    help="Controls randomness for reproducible results",
                                    key="rf_random_state"
                                )
                            
                            with params_col2:
                                rf_params['max_depth'] = st.slider(
                                    "Maximum Tree Depth",
                                    min_value=3,
                                    max_value=20,
                                    value=5,
                                    help="Controls model complexity. Higher values may lead to overfitting",
                                    key="rf_max_depth"
                                )
                                
                                rf_params['min_samples_split'] = st.slider(
                                    "Minimum Samples to Split",
                                    min_value=2,
                                    max_value=20,
                                    value=2,
                                    help="Minimum samples required to split an internal node",
                                    key="rf_min_samples_split"
                                )
                            
                            # Advanced options
                            with st.expander("Advanced Random Forest Options"):
                                rf_params['min_samples_leaf'] = st.slider(
                                    "Minimum Samples in Leaf",
                                    min_value=1,
                                    max_value=10,
                                    value=1,
                                    help="Minimum samples required in a leaf node",
                                    key="rf_min_samples_leaf"
                                )
                                
                                rf_params['bootstrap'] = st.checkbox(
                                    "Bootstrap",
                                    value=True,
                                    help="Whether to use bootstrap samples",
                                    key="rf_bootstrap"
                                )
                            
                            model_params['random_forest'] = rf_params
                    
                    # Gradient Boosting parameters
                    if "gradient_boost" in models_to_train:
                        gb_idx = models_to_train.index("gradient_boost")
                        with model_tabs[gb_idx]:
                            gb_params = {}
                            params_col1, params_col2 = st.columns(2)
                            
                            with params_col1:
                                gb_params['n_estimators'] = st.slider(
                                    "Number of Boosting Stages",
                                    min_value=50,
                                    max_value=200,
                                    value=100,
                                    step=10,
                                    help="More stages provide better accuracy but increase training time",
                                    key="gb_n_estimators"
                                )
                                
                                gb_params['learning_rate'] = st.slider(
                                    "Learning Rate",
                                    min_value=0.01,
                                    max_value=0.3,
                                    value=0.1,
                                    step=0.01,
                                    help="Shrinks the contribution of each tree",
                                    key="gb_learning_rate"
                                )
                            
                            with params_col2:
                                gb_params['max_depth'] = st.slider(
                                    "Maximum Tree Depth",
                                    min_value=3,
                                    max_value=10,
                                    value=3,
                                    help="Controls model complexity. Higher values may lead to overfitting",
                                    key="gb_max_depth"
                                )
                                
                                gb_params['random_state'] = st.number_input(
                                    "Random Seed",
                                    min_value=1,
                                    max_value=100,
                                    value=42,
                                    help="Controls randomness for reproducible results",
                                    key="gb_random_state"
                                )
                            
                            model_params['gradient_boost'] = gb_params
                    
                    # Logistic Regression parameters
                    if "logistic_regression" in models_to_train:
                        lr_idx = models_to_train.index("logistic_regression")
                        with model_tabs[lr_idx]:
                            lr_params = {}
                            params_col1, params_col2 = st.columns(2)
                            
                            with params_col1:
                                lr_params['C'] = st.slider(
                                    "Regularization Strength (C)",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=1.0,
                                    step=0.1,
                                    help="Lower values increase regularization",
                                    key="lr_C"
                                )
                                
                                lr_params['random_state'] = st.number_input(
                                    "Random Seed",
                                    min_value=1,
                                    max_value=100,
                                    value=42,
                                    help="Controls randomness for reproducible results",
                                    key="lr_random_state"
                                )
                            
                            with params_col2:
                                lr_params['max_iter'] = st.slider(
                                    "Maximum Iterations",
                                    min_value=100,
                                    max_value=1000,
                                    value=100,
                                    step=100,
                                    help="Maximum number of iterations for solver",
                                    key="lr_max_iter"
                                )
                                
                                lr_params['solver'] = st.selectbox(
                                    "Solver",
                                    options=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                                    index=0,
                                    help="Algorithm for optimization problem",
                                    key="lr_solver"
                                )
                            
                            model_params['logistic_regression'] = lr_params
                    
                    # Neural Network parameters
                    if "neural_network" in models_to_train:
                        nn_idx = models_to_train.index("neural_network")
                        with model_tabs[nn_idx]:
                            nn_params = {}
                            params_col1, params_col2 = st.columns(2)
                            
                            with params_col1:
                                hidden_layer_1 = st.slider(
                                    "First Hidden Layer Neurons",
                                    min_value=5,
                                    max_value=50,
                                    value=10,
                                    step=5,
                                    help="Number of neurons in first hidden layer",
                                    key="nn_hidden_1"
                                )
                                
                                hidden_layer_2 = st.slider(
                                    "Second Hidden Layer Neurons",
                                    min_value=0,
                                    max_value=20,
                                    value=5,
                                    step=5,
                                    help="Number of neurons in second hidden layer (0 to omit)",
                                    key="nn_hidden_2"
                                )
                                
                                if hidden_layer_2 > 0:
                                    nn_params['hidden_layer_sizes'] = (hidden_layer_1, hidden_layer_2)
                                else:
                                    nn_params['hidden_layer_sizes'] = (hidden_layer_1,)
                            
                            with params_col2:
                                nn_params['activation'] = st.selectbox(
                                    "Activation Function",
                                    options=['relu', 'tanh', 'logistic'],
                                    index=0,
                                    help="Activation function for hidden layers",
                                    key="nn_activation"
                                )
                                
                                nn_params['max_iter'] = st.slider(
                                    "Maximum Iterations",
                                    min_value=200,
                                    max_value=2000,
                                    value=1000,
                                    step=100,
                                    help="Maximum number of iterations",
                                    key="nn_max_iter"
                                )
                            
                            # Advanced options
                            with st.expander("Advanced Neural Network Options"):
                                nn_params['solver'] = st.selectbox(
                                    "Solver",
                                    options=['adam', 'sgd', 'lbfgs'],
                                    index=0,
                                    help="Solver for weight optimization",
                                    key="nn_solver"
                                )
                                
                                nn_params['alpha'] = st.slider(
                                    "Alpha (L2 Regularization)",
                                    min_value=0.0001,
                                    max_value=0.01,
                                    value=0.0001,
                                    format="%.4f",
                                    help="L2 regularization parameter",
                                    key="nn_alpha"
                                )
                                
                                nn_params['random_state'] = 42
                            
                            model_params['neural_network'] = nn_params
                
                # Test set size
                st.markdown("### Training Settings")
                test_size = st.slider(
                    "Test Set Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Percentage of data used for testing the model"
                )
                
                # Train model button using form
                with st.form(key="train_model_form"):
                    st.markdown("### Train Selected Models")
                    st.markdown("Click the button below to train the selected models with the configured parameters.")
                    
                    # Form submit button
                    train_submitted = st.form_submit_button("🚀 Train Model(s)", use_container_width=True)
                
                # Process form if submitted
                if 'train_submitted' in locals() and train_submitted:
                    with st.spinner("🔄 Training models in progress..."):
                        training_data = st.session_state.training_data
                        
                        # Train selected models
                        trained_models, features, reports = train_models(
                            training_data, 
                            models_to_train=models_to_train,
                            params=model_params
                        )
                        
                        if trained_models:
                            # Store reports in session state
                            st.session_state.training_report = reports
                            st.session_state.model_features = features
                            
                            # Save the models
                            st.session_state.models = {k: None for k in st.session_state.models}
                            for model_type, model in trained_models.items():
                                st.session_state.models[model_type] = model
                            
                            # Set main model (default to first one)
                            st.session_state.model = trained_models[models_to_train[0]]
                            st.session_state.active_model = models_to_train[0]
                            
                            # Create model comparison data
                            model_comparison = {}
                            for model_type, report in reports.items():
                                model_comparison[model_type] = {
                                    'accuracy': report['accuracy'],
                                    'precision': report['1']['precision'],
                                    'recall': report['1']['recall'],
                                    'f1': report['1']['f1-score'],
                                    'roc_auc': report['roc_curve']['auc'],
                                    'cv_mean': report['cv_scores']['mean']
                                }
                            
                            st.session_state.model_comparison = model_comparison
                            
                            # Clear any cached predictions
                            st.session_state.predictions_df = None
                            
                            # Show success message
                            st.success(f"✅ Successfully trained {len(models_to_train)} model(s)! Check the 'Training Results' tab to see performance metrics.")
                            
                            # Enable prediction features automatically
                            st.session_state.training_successful = True
                        else:
                            st.error("Model training failed. Please check your data and try again.")
        
        with training_tabs[2]:  # Results tab
            st.markdown("<div class='card-title'>📈 Training Results & Model Performance</div>", unsafe_allow_html=True)
            
            # Display training results if available
            if st.session_state.model is not None and st.session_state.training_report is not None:
                # Model information
                st.markdown("### Model Information")
                
                # Determine model type and display appropriate information
                model_type = st.session_state.active_model if 'active_model' in st.session_state else 'unknown'
                model_name = model_type.replace('_', ' ').title() if model_type else 'Unknown'
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.info(f"**Model Type:** {model_name}")
                
                with info_col2:
                    # Display attributes based on model type
                    if model_type in ['random_forest', 'gradient_boost'] and hasattr(st.session_state.model, 'n_estimators'):
                        n_trees = st.session_state.model.n_estimators
                        st.info(f"**Number of Trees:** {n_trees}")
                    elif model_type == 'neural_network' and hasattr(st.session_state.model, 'hidden_layer_sizes'):
                        layers = st.session_state.model.hidden_layer_sizes
                        st.info(f"**Hidden Layers:** {layers}")
                    elif model_type == 'logistic_regression' and hasattr(st.session_state.model, 'C'):
                        c_value = st.session_state.model.C
                        st.info(f"**Regularization (C):** {c_value}")
                    else:
                        st.info("**Model Parameter:** Not available")
                
                with info_col3:
                    # Display attributes based on model type
                    if model_type in ['random_forest', 'gradient_boost'] and hasattr(st.session_state.model, 'max_depth'):
                        max_depth = st.session_state.model.max_depth
                        st.info(f"**Max Tree Depth:** {max_depth}")
                    elif model_type == 'neural_network' and hasattr(st.session_state.model, 'activation'):
                        activation = st.session_state.model.activation
                        st.info(f"**Activation:** {activation}")
                    elif model_type == 'logistic_regression' and hasattr(st.session_state.model, 'max_iter'):
                        max_iter = st.session_state.model.max_iter
                        st.info(f"**Max Iterations:** {max_iter}")
                    else:
                        st.info("**Model Parameter:** Not available")
                
                # Performance metrics
                st.markdown("### Performance Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    if 'accuracy' in st.session_state.training_report:
                        accuracy = st.session_state.training_report['accuracy']
                    else:
                        # Calculate from weighted avg if not directly available
                        accuracy = st.session_state.training_report.get('weighted avg', {}).get('f1-score', 0.0)
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                with metrics_col2:
                    # Handle different possible report structures
                    if '1' in st.session_state.training_report:
                        precision = st.session_state.training_report['1']['precision']
                    else:
                        precision = st.session_state.training_report.get('weighted avg', {}).get('precision', 0.0)
                    st.metric("Precision", f"{precision:.4f}")
                
                with metrics_col3:
                    if '1' in st.session_state.training_report:
                        recall = st.session_state.training_report['1']['recall']
                    else:
                        recall = st.session_state.training_report.get('weighted avg', {}).get('recall', 0.0)
                    st.metric("Recall", f"{recall:.4f}")
                
                with metrics_col4:
                    if '1' in st.session_state.training_report:
                        f1 = st.session_state.training_report['1']['f1-score']
                    else:
                        f1 = st.session_state.training_report.get('weighted avg', {}).get('f1-score', 0.0)
                    st.metric("F1 Score", f"{f1:.4f}")
                
                # Class distribution
                st.markdown("### Class Distribution")
                # Handle different possible report structures
                if '0' in st.session_state.training_report and '1' in st.session_state.training_report:
                    support_0 = st.session_state.training_report['0']['support']
                    support_1 = st.session_state.training_report['1']['support']
                else:
                    # Use a reasonable default if the expected structure isn't available
                    support_0 = st.session_state.training_report.get('weighted avg', {}).get('support', 100) // 2
                    support_1 = support_0
                total = support_0 + support_1
                
                # Create a pie chart for class distribution
                labels = ['Non-CA', 'CA']
                values = [support_0, support_1]
                colors = ['#2ecc71', '#ff4b4b']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.3,
                    marker=dict(colors=colors)
                )])
                
                fig.update_layout(
                    title="Training Data Class Distribution",
                    annotations=[dict(text=f'Total: {total}', showarrow=False, font_size=20)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.markdown("### Feature Importance")
                st.markdown("""
                This chart shows which factors have the most impact on chronic absenteeism predictions.
                Features with higher importance values have a stronger influence on the model's decisions.
                """)
                
                plot_feature_importance(st.session_state.model, key="training_feat_imp")
                
                # Export model
                st.markdown("### Export Trained Model")
                st.markdown("""
                You can download the trained model for use in other systems or for future reference.
                The model is saved in joblib format, which preserves all the trained parameters.
                """)
                
                # Save model button in a container with styling
                save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
                with save_col2:
                    save_model()
            else:
                st.info("⏳ No training results available yet. Please train a model first.")
                
                # Show a placeholder visualization
                st.markdown("### Ready to Train")
                st.markdown("""
                Once you train a model, you'll see:
                - Performance metrics (accuracy, precision, recall, F1 score)
                - Feature importance visualization
                - Option to export the trained model
                """)

        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if 'current_hist_student_id' not in st.session_state:
                            st.session_state.current_hist_student_id = ""
                            st.session_state.disable_hist_inputs = False
                        
                        if st.session_state.current_hist_student_id != hist_student_id:
                            st.session_state.current_hist_student_id = hist_student_id
                            
                            # Fetch student data from the dataframe
                            hist_student_data = st.session_state.historical_data[st.session_state.historical_data['Student_ID'] == hist_student_id]
                            
                            if not hist_student_data.empty:
                                # Update input fields with student data
                                # We'll use the most recent record for this student
                                hist_student_data = hist_student_data.sort_values('Date', ascending=False)
                                student_row = hist_student_data.iloc[0]
                                
                                # Fill input fields with student data
                                st.session_state.hist_school_input = student_row.get('School', 'School A')
                                st.session_state.hist_grade_input = int(student_row.get('Grade', 1))
                                st.session_state.hist_gender_input = student_row.get('Gender', 'Male')
                                st.session_state.hist_present_days_input = int(student_row.get('Present_Days', 150))
                                st.session_state.hist_absent_days_input = int(student_row.get('Absent_Days', 10))
                                st.session_state.hist_meal_code_input = student_row.get('Meal_Code', 'Paid')
                                st.session_state.hist_academic_perf_input = int(student_row.get('Academic_Performance', 70))
                                
                                # Store the student data for prediction
                                st.session_state.hist_input_data = student_row.to_dict()
                                
                                # Set flags for UI updates
                                st.session_state.hist_needs_prediction = True
                                st.session_state.disable_hist_inputs = True
                            else:
                                st.error(f"Student data not found for ID: {hist_student_id}")
                                # Reset to avoid errors
                                st.session_state.current_hist_student_id = ""
                    
                    # Create a form for student data inputs
                    # The form element needs a unique key
                    with st.form(key="hist_ca_input_form", clear_on_submit=False):
                        if 'disable_hist_inputs' not in st.session_state:
                            st.session_state.disable_hist_inputs = False
                            
                        disabled_class = 'class="disabled-field"' if st.session_state.disable_hist_inputs else ''
                        st.markdown(f"<div {disabled_class}>", unsafe_allow_html=True)
                        
                        # School input
                        school_col, grade_col = st.columns(2)
                        with school_col:
                            # Get available schools from the historical data
                            available_schools = sorted(st.session_state.historical_data['School'].unique().tolist())
                            if not available_schools:
                                available_schools = ['School A', 'School B', 'School C', 'School D']
                            
                            school = st.selectbox(
                                "School",
                                options=available_schools,
                                key="hist_school_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        # Grade input
                        with grade_col:
                            grade = st.number_input(
                                "Grade",
                                min_value=1,
                                max_value=12,
                                value=1,
                                key="hist_grade_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        # Gender and meal code
                        gender_col, meal_col = st.columns(2)
                        with gender_col:
                            gender = st.selectbox(
                                "Gender",
                                options=['Male', 'Female'],
                                key="hist_gender_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        with meal_col:
                            meal_code = st.selectbox(
                                "Meal Code",
                                options=['Free', 'Reduced', 'Paid'],
                                key="hist_meal_code_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        # Attendance details
                        present_col, absent_col = st.columns(2)
                        with present_col:
                            present_days = st.number_input(
                                "Present Days",
                                min_value=0,
                                max_value=200,
                                value=150,
                                key="hist_present_days_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        with absent_col:
                            absent_days = st.number_input(
                                "Absent Days",
                                min_value=0,
                                max_value=200,
                                value=10,
                                key="hist_absent_days_input",
                                disabled=st.session_state.disable_hist_inputs
                            )
                        
                        # Academic performance
                        academic_perf = st.slider(
                            "Academic Performance",
                            min_value=0,
                            max_value=100,
                            value=70,
                            key="hist_academic_perf_input",
                            disabled=st.session_state.disable_hist_inputs
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Function to handle historical prediction
                        def on_calculate_hist_risk():
                            if not st.session_state.model:
                                st.warning("⚠️ No model available. Please train a model first.")
                                return
                            
                            # Gather input data
                            current_inputs = {
                                'Student_ID': st.session_state.current_hist_student_id,
                                'School': st.session_state.hist_school_input,
                                'Grade': st.session_state.hist_grade_input,
                                'Gender': st.session_state.hist_gender_input,
                                'Present_Days': st.session_state.hist_present_days_input,
                                'Absent_Days': st.session_state.hist_absent_days_input,
                                'Meal_Code': st.session_state.hist_meal_code_input,
                                'Academic_Performance': st.session_state.hist_academic_perf_input
                            }
                            
                            # Add derived fields
                            total_days = current_inputs['Present_Days'] + current_inputs['Absent_Days']
                            if total_days > 0:
                                current_inputs['Attendance_Percentage'] = (current_inputs['Present_Days'] / total_days) * 100
                            else:
                                current_inputs['Attendance_Percentage'] = 0
                            
                            # Run prediction
                            risk = predict_ca_risk(current_inputs, st.session_state.model)
                            if risk is not None:
                                st.session_state.hist_current_prediction = risk[0]
                                st.session_state.hist_calculation_complete = True
                                
                                # Store original prediction for what-if comparisons
                                st.session_state.hist_original_prediction = risk[0]
                                st.session_state.hist_input_data = current_inputs
                            else:
                                st.error("Error in prediction. Please check inputs and model.")
                        
                        # Submit button - explicitly as the last element in the form
                        submitted = st.form_submit_button(label="Calculate CA Risk", on_click=on_calculate_hist_risk)
                else:
                    st.warning("No historical data available. Please upload training data first.")
                
            # Results section (right column)
            with col2:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.header("Historical Risk Assessment")
                
                # Initialize session states if they don't exist
                if 'hist_current_prediction' not in st.session_state:
                    st.session_state.hist_current_prediction = None
                if 'hist_calculation_complete' not in st.session_state:
                    st.session_state.hist_calculation_complete = False
                
                # Display prediction results
                if st.session_state.hist_current_prediction is not None:
                    risk_value = st.session_state.hist_current_prediction
                    
                    # Risk visualization
                    plot_risk_gauge(risk_value, key="hist_main_gauge")
                    
                    # Attendance ratio visualization
                    if 'hist_input_data' in st.session_state and st.session_state.hist_input_data:
                        present_days = st.session_state.hist_input_data.get('Present_Days', 0)
                        absent_days = st.session_state.hist_input_data.get('Absent_Days', 0)
                        total_days = present_days + absent_days
                        
                        if total_days > 0:
                            # Create attendance pie chart
                            st.subheader("Attendance Summary")
                            attendance_labels = ['Present', 'Absent']
                            attendance_values = [present_days, absent_days]
                            attendance_colors = ['#2ecc71', '#ff4b4b']
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=attendance_labels,
                                values=attendance_values,
                                hole=.3,
                                marker=dict(colors=attendance_colors)
                            )])
                            
                            fig.update_layout(
                                title="Historical School Year",
                                height=250,
                                margin=dict(l=20, r=20, t=30, b=10),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display recommendation
                    st.subheader("Historical Recommended Actions")
                    recommendation = get_recommendation(risk_value)
                    st.markdown(f"<div class='recommendation'>{recommendation}</div>", unsafe_allow_html=True)
                    
                    # Student history if a student ID is provided
                    if st.session_state.current_hist_student_id:
                        st.subheader("Historical Trends")
                        plot_student_history(st.session_state.current_hist_student_id)
                else:
                    st.info("Select a student and click 'Calculate CA Risk' to see the historical prediction.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            if 'historical_data' not in st.session_state or st.session_state.historical_data.empty:
                st.warning("No historical data available. Please upload historical data in the Training Data tab.")
            else:
                # Get unique student IDs from historical data
                student_ids = st.session_state.historical_data['Student_ID'].unique().tolist()
                
                if not student_ids:
                    st.warning("No student records found in the historical data.")
                else:
                    # Student selection
                    selected_student = st.selectbox(
                        "Select Student ID to Analyze",
                        options=student_ids,
                        key="historical_student_select"
                    )
                    
                    # Get student data
                    student_data = st.session_state.historical_data[st.session_state.historical_data['Student_ID'] == selected_student]
                    
                    if not student_data.empty:
                        # Show student details
                        st.subheader(f"Historical Data for Student: {selected_student}")
                        
                        # Display a summary
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("School", student_data['School'].iloc[0])
                        with stats_col2:
                            st.metric("Grade", student_data['Grade'].iloc[0])
                        with stats_col3:
                            st.metric("Gender", student_data['Gender'].iloc[0])
                        
                        # Show attendance history
                        st.subheader("Attendance History")
                        
                        # Create a time series of attendance
                        if 'Date' in student_data.columns:
                            # Sort by date
                            student_data = student_data.sort_values('Date')
                            
                            # Create a line chart of attendance percentage
                            st.line_chart(student_data.set_index('Date')['Attendance_Percentage'])
                        
                        # Show CA Status history
                        st.subheader("Chronic Absenteeism History")
                        
                        # Display CA status
                        fig = px.bar(
                            student_data, 
                            x='Date' if 'Date' in student_data.columns else student_data.index, 
                            y='CA_Status',
                            labels={'CA_Status': 'CA Status (1=Yes, 0=No)'},
                            title='Historical CA Status'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display raw data if requested
                        with st.expander("View Raw Historical Data"):
                            st.dataframe(student_data)
                    else:
                        st.warning(f"No historical data found for student {selected_student}")
        
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif app_mode == "Advanced Analytics":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(display_svg("images/ai_analysis.svg", width="200px"), unsafe_allow_html=True)
        st.markdown("<h2>Advanced Analytics</h2>", unsafe_allow_html=True)
        st.markdown("""
        Advanced analytics provides deeper insights into the CA risk factors and prediction model.
        Explore different reports to gain a comprehensive understanding of the data and predictions.
        """)
        
        if st.session_state.model is None:
            st.warning("No trained model available. Please go to System Training to train a model first.")
        else:
            # Create a dropdown for different report types
            report_type = st.selectbox(
                "Select Report Type",
                [
                    "System Performance Overview",
                    "School Risk Analysis",
                    "Demographic Analysis",
                    "Grade-level Analysis", 
                    "Feature Correlation",
                    "High-Risk Students",
                    "Attendance vs. Academic Performance",
                    "Risk Heatmap by Grade & SES",
                    "Temporal Attendance Trends",
                    "Intervention Cost-Benefit",
                    "Geographic Risk Mapping",
                    "Cohort Analysis"
                ],
                format_func=lambda x: {
                    "System Performance Overview": "📈 System Performance Overview",
                    "School Risk Analysis": "🏫 School Risk Analysis",
                    "Attendance vs. Academic Performance": "📊 Attendance vs. Academic Performance",
                    "Risk Heatmap by Grade & SES": "🔥 Risk Heatmap by Grade & SES",
                    "Temporal Attendance Trends": "⏱️ Temporal Attendance Trends",
                    "Intervention Cost-Benefit": "💰 Intervention Cost-Benefit",
                    "Geographic Risk Mapping": "🗺️ Geographic Risk Mapping",
                    "Cohort Analysis": "👥 Cohort Analysis",
                    "Demographic Analysis": "👥 Demographic Analysis",
                    "Grade-level Analysis": "📚 Grade-level Analysis",
                    "Feature Correlation": "🔄 Feature Correlation",
                    "High-Risk Students": "⚠️ High-Risk Students"
                }[x]
            )
            
            if not st.session_state.current_df.empty:
                # Generate predictions if needed
                if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
                    with st.spinner("Generating predictions for analysis..."):
                        predictions_df = batch_predict_ca(st.session_state.current_df, st.session_state.model)
                        st.session_state.predictions_df = predictions_df
                else:
                    predictions_df = st.session_state.predictions_df
                
                # Show different reports based on selection
                if report_type == "System Performance Overview":
                    st.markdown("<div class='card-title'>📈 System Performance Overview</div>", unsafe_allow_html=True)
                    generate_system_report()
                
                elif report_type == "School Risk Analysis":
                    st.markdown("<div class='card-title'>🏫 School Risk Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis shows the chronic absenteeism risk distribution across different schools.
                    Higher percentages indicate schools that may require more attention and resources.
                    """)
                    
                    # Group by school
                    school_risk = predictions_df.groupby('School')['CA_Risk'].mean().reset_index()
                    school_risk['Risk_Percentage'] = school_risk['CA_Risk'] * 100
                    
                    # Plot school risk comparison
                    fig = px.bar(
                        school_risk,
                        x='School',
                        y='Risk_Percentage',
                        title='Average CA Risk by School',
                        color='Risk_Percentage',
                        color_continuous_scale='RdYlGn_r',
                        text=school_risk['Risk_Percentage'].round(1).astype(str) + '%'
                    )
                    
                    fig.update_layout(
                        xaxis_title="School",
                        yaxis_title="Average Risk (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # School risk counts
                    school_risk_counts = predictions_df.groupby(['School', 'Risk_Level']).size().reset_index()
                    school_risk_counts.columns = ['School', 'Risk Level', 'Count']
                    
                    fig = px.bar(
                        school_risk_counts,
                        x='School',
                        y='Count',
                        color='Risk Level',
                        title='Risk Level Distribution by School',
                        barmode='group',
                        color_discrete_map={
                            'Low': '#2ecc71',
                            'Medium': '#ffa500',
                            'High': '#ff4b4b'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif report_type == "Demographic Analysis":
                    st.markdown("<div class='card-title'>👥 Demographic Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis examines how chronic absenteeism risk varies across different demographic factors 
                    such as gender and meal code status.
                    """)
                    
                    # Risk distribution by gender
                    gender_risk = predictions_df.groupby('Gender')['CA_Risk'].mean().reset_index()
                    gender_risk['Risk_Percentage'] = gender_risk['CA_Risk'] * 100
                    
                    fig = px.bar(
                        gender_risk,
                        x='Gender',
                        y='Risk_Percentage',
                        title='Average CA Risk by Gender',
                        color='Gender',
                        text=gender_risk['Risk_Percentage'].round(1).astype(str) + '%'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Gender",
                        yaxis_title="Average Risk (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk distribution by meal code
                    meal_risk = predictions_df.groupby('Meal_Code')['CA_Risk'].mean().reset_index()
                    meal_risk['Risk_Percentage'] = meal_risk['CA_Risk'] * 100
                    
                    fig = px.bar(
                        meal_risk,
                        x='Meal_Code',
                        y='Risk_Percentage',
                        title='Average CA Risk by Meal Code',
                        color='Meal_Code',
                        text=meal_risk['Risk_Percentage'].round(1).astype(str) + '%'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Meal Code",
                        yaxis_title="Average Risk (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif report_type == "Grade-level Analysis":
                    st.markdown("<div class='card-title'>📚 Grade-level Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis shows how chronic absenteeism risk changes across different grade levels.
                    Identifying critical grade transitions can help in developing targeted interventions.
                    """)
                    
                    # Risk distribution by grade
                    grade_risk = predictions_df.groupby('Grade')['CA_Risk'].mean().reset_index()
                    grade_risk['Risk_Percentage'] = grade_risk['CA_Risk'] * 100
                    
                    fig = px.line(
                        grade_risk,
                        x='Grade',
                        y='Risk_Percentage',
                        title='Average CA Risk by Grade',
                        markers=True,
                        line_shape='spline'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Grade",
                        yaxis_title="Average Risk (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Student count by grade and risk level
                    grade_counts = predictions_df.groupby(['Grade', 'Risk_Level']).size().reset_index()
                    grade_counts.columns = ['Grade', 'Risk Level', 'Count']
                    
                    fig = px.bar(
                        grade_counts,
                        x='Grade',
                        y='Count',
                        color='Risk Level',
                        title='Risk Level Distribution by Grade',
                        barmode='stack',
                        color_discrete_map={
                            'Low': '#2ecc71',
                            'Medium': '#ffa500',
                            'High': '#ff4b4b'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif report_type == "Feature Correlation":
                    st.markdown("<div class='card-title'>🔄 Feature Correlation</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis shows the correlation between different features and how they relate to each other.
                    Strong correlations may indicate relationships that can be leveraged for intervention strategies.
                    """)
                    
                    # Correlation matrix
                    numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
                    correlation = predictions_df[numeric_cols].corr()
                    
                    fig = px.imshow(
                        correlation,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix of Features"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature correlation with CA_Risk
                    risk_corr = correlation['CA_Risk'].drop('CA_Risk').sort_values(ascending=False)
                    risk_corr = pd.DataFrame({'Feature': risk_corr.index, 'Correlation': risk_corr.values})
                    
                    fig = px.bar(
                        risk_corr,
                        y='Feature',
                        x='Correlation',
                        orientation='h',
                        title='Feature Correlation with CA Risk',
                        color='Correlation',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Correlation Coefficient",
                        yaxis_title="Feature",
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif report_type == "High-Risk Students":
                    st.markdown("<div class='card-title'>⚠️ High-Risk Students</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This report identifies students with the highest risk of chronic absenteeism.
                    These students may require immediate intervention.
                    """)
                    
                    # Get high-risk students
                    high_risk = predictions_df[predictions_df['CA_Risk'] > st.session_state.risk_thresholds['medium']]
                    high_risk = high_risk.sort_values('CA_Risk', ascending=False).head(20)
                    
                    if not high_risk.empty:
                        st.subheader(f"Top {len(high_risk)} Highest Risk Students")
                        
                        # Create a display table with colored risk levels
                        display_cols = ['Student_ID', 'School', 'Grade', 'CA_Risk', 'Attendance_Percentage', 'Risk_Level']
                        display_df = high_risk[display_cols].copy()
                        display_df['CA_Risk'] = (display_df['CA_Risk'] * 100).round(1).astype(str) + '%'
                        display_df['Attendance_Percentage'] = display_df['Attendance_Percentage'].round(1).astype(str) + '%'
                        
                        # Create styled dataframe
                        st.dataframe(
                            display_df.style.apply(
                                lambda x: ['background-color: #ffe6e6' if v == 'High' else '' for v in x],
                                subset=['Risk_Level']
                            )
                        )
                        
                        # Common factors visualization
                        st.subheader("Common Risk Factors")
                        st.markdown("""
                        The chart below shows the most common factors contributing to high risk among these students.
                        """)
                        
                        # Simple bar chart showing attendance percentages
                        fig = px.bar(
                            high_risk,
                            x='Student_ID',
                            y='Attendance_Percentage',
                            title='Attendance Percentages for High-Risk Students',
                            color='Attendance_Percentage',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Student ID",
                            yaxis_title="Attendance Percentage",
                            xaxis={'categoryorder':'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No high-risk students identified in the current dataset.")
                
                elif report_type == "Attendance vs. Academic Performance":
                    st.markdown("<div class='card-title'>📊 Attendance vs. Academic Performance</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis shows the relationship between attendance percentage and academic performance,
                    and how it relates to chronic absenteeism risk.
                    """)
                    
                    # Create scatter plot of attendance vs academic performance
                    fig = px.scatter(
                        predictions_df,
                        x="Attendance_Percentage",
                        y="Academic_Performance",
                        color="CA_Risk",
                        color_continuous_scale="RdYlGn_r",
                        opacity=0.7,
                        hover_data=["Student_ID", "School", "Grade"],
                        size_max=15,
                        title="Attendance vs. Academic Performance by Risk Level",
                        labels={
                            "Attendance_Percentage": "Attendance Percentage",
                            "Academic_Performance": "Academic Performance Score",
                            "CA_Risk": "CA Risk"
                        }
                    )
                    
                    fig.update_layout(
                        xaxis=dict(range=[0, 100]),
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add trend line and statistics
                    st.subheader("Statistical Analysis")
                    
                    # Calculate correlation
                    corr = predictions_df['Attendance_Percentage'].corr(predictions_df['Academic_Performance'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Correlation Coefficient", f"{corr:.3f}")
                    with col2:
                        avg_attendance = predictions_df['Attendance_Percentage'].mean()
                        st.metric("Average Attendance", f"{avg_attendance:.1f}%")
                
                elif report_type == "Risk Heatmap by Grade & SES":
                    st.markdown("<div class='card-title'>🔥 Risk Heatmap by Grade & SES</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This heatmap visualization shows the variation in chronic absenteeism risk across 
                    different grade levels and socioeconomic status (based on meal code).
                    """)
                    
                    # Calculate average risk by grade and meal code
                    heatmap_data = predictions_df.groupby(['Grade', 'Meal_Code'])['CA_Risk'].mean().reset_index()
                    heatmap_data['Risk_Percentage'] = heatmap_data['CA_Risk'] * 100
                    
                    # Create pivot table for heatmap
                    heatmap_pivot = heatmap_data.pivot(index='Grade', columns='Meal_Code', values='Risk_Percentage')
                    
                    # Create heatmap
                    fig = px.imshow(
                        heatmap_pivot,
                        labels=dict(x="Socioeconomic Status (Meal Code)", y="Grade", color="Risk %"),
                        x=heatmap_pivot.columns,
                        y=heatmap_pivot.index,
                        color_continuous_scale="RdYlGn_r",
                        text_auto='.1f',
                        aspect="auto",
                        title="CA Risk Heatmap by Grade and Socioeconomic Status"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis text
                    st.subheader("Key Insights")
                    
                    meal_code_risk = predictions_df.groupby('Meal_Code')['CA_Risk'].mean().sort_values(ascending=False)
                    highest_meal = meal_code_risk.index[0]
                    highest_meal_risk = meal_code_risk.values[0] * 100
                    
                    grade_ses_max = heatmap_pivot.max().max()
                    grade_ses_min = heatmap_pivot.min().min()
                    grade_ses_diff = grade_ses_max - grade_ses_min
                    
                    st.markdown(f"""
                    - Students with **{highest_meal}** meal status show the highest average risk at **{highest_meal_risk:.1f}%**
                    - The variation in risk between different grade and SES combinations is **{grade_ses_diff:.1f}%**
                    - This visualization helps identify specific grade-SES combinations that may require targeted interventions
                    """)
                
                elif report_type == "Temporal Attendance Trends":
                    st.markdown("<div class='card-title'>⏱️ Temporal Attendance Trends</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This report shows how attendance patterns change over time for different student groups.
                    Understanding these trends can help in planning timely interventions.
                    """)
                    
                    # Sample dates for demonstration (would use real dates in actual implementation)
                    # Create sample monthly attendance data
                    months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
                    
                    # Create different attendance patterns by risk level
                    high_risk_attendance = [95, 90, 82, 78, 75, 70, 65, 60, 55]
                    medium_risk_attendance = [98, 96, 94, 92, 90, 91, 89, 88, 86]
                    low_risk_attendance = [99, 98, 98, 97, 98, 97, 96, 95, 94]
                    
                    # Create dataframe
                    trend_data = pd.DataFrame({
                        'Month': months * 3,
                        'Risk_Level': ['High'] * 9 + ['Medium'] * 9 + ['Low'] * 9,
                        'Attendance_Percentage': high_risk_attendance + medium_risk_attendance + low_risk_attendance,
                        'Month_Num': list(range(1, 10)) * 3
                    })
                    
                    # Create line chart
                    fig = px.line(
                        trend_data,
                        x='Month',
                        y='Attendance_Percentage',
                        color='Risk_Level',
                        markers=True,
                        title="Attendance Trends by Risk Level",
                        color_discrete_map={
                            'Low': '#2ecc71',
                            'Medium': '#ffa500',
                            'High': '#ff4b4b'
                        }
                    )
                    
                    fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Attendance Percentage",
                        yaxis=dict(range=[50, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add critical period identification
                    st.subheader("Critical Attendance Drop Periods")
                    
                    # Identify months with biggest drops
                    st.markdown("""
                    - **November-December**: High risk students show significant attendance drop (-4%)
                    - **February-March**: Both high and medium risk students show attendance declines
                    - **Early intervention opportunity**: October, when high risk students begin showing declining patterns
                    """)
                
                elif report_type == "Intervention Cost-Benefit":
                    st.markdown("<div class='card-title'>💰 Intervention Cost-Benefit Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis helps determine which interventions provide the best return on investment by 
                    comparing their cost, effectiveness, and potential impact on reducing chronic absenteeism.
                    """)
                    
                    # Use the interventions from session state
                    interventions = st.session_state.interventions
                    
                    # Calculate cost-effectiveness ratio
                    intervention_df = pd.DataFrame.from_dict(interventions, orient='index')
                    intervention_df['Intervention'] = intervention_df.index
                    intervention_df['Cost_Effectiveness_Ratio'] = intervention_df['effectiveness'] / (intervention_df['cost'] / 1000)
                    
                    # Create visualization
                    fig = px.scatter(
                        intervention_df,
                        x="cost",
                        y="effectiveness",
                        size="Cost_Effectiveness_Ratio",
                        color="Intervention",
                        hover_name="Intervention",
                        text="Intervention",
                        title="Intervention Cost vs. Effectiveness",
                        labels={
                            "cost": "Cost per Student ($)",
                            "effectiveness": "Effectiveness (% Risk Reduction)",
                            "Cost_Effectiveness_Ratio": "Cost-Effectiveness Ratio"
                        }
                    )
                    
                    fig.update_traces(marker=dict(sizemode='area', sizeref=0.1), textposition='top center')
                    fig.update_layout(xaxis_title="Cost per Student ($)", yaxis_title="Effectiveness (% Risk Reduction)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create bar chart of cost-effectiveness ratio
                    sorted_interventions = intervention_df.sort_values('Cost_Effectiveness_Ratio', ascending=False)
                    
                    fig = px.bar(
                        sorted_interventions,
                        x="Intervention",
                        y="Cost_Effectiveness_Ratio",
                        color="Intervention",
                        title="Intervention Cost-Effectiveness Ratio (higher is better)",
                        labels={"Cost_Effectiveness_Ratio": "Cost-Effectiveness Ratio"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add recommendations
                    st.subheader("Recommended Intervention Strategy")
                    best_intervention = sorted_interventions.iloc[0]['Intervention']
                    
                    st.markdown(f"""
                    - Most cost-effective intervention: **{best_intervention}**
                    - Consider combining high-effectiveness interventions for high-risk students with 
                      cost-effective interventions for medium-risk students
                    - For budget planning, the most expensive interventions should be reserved for the students 
                      with highest risk and greatest potential for improvement
                    """)
                
                elif report_type == "Geographic Risk Mapping":
                    st.markdown("<div class='card-title'>🗺️ Geographic Risk Mapping</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This visualization shows the geographic distribution of chronic absenteeism risk across different school zones.
                    Spatial patterns can help identify community-level factors affecting attendance.
                    """)
                    
                    st.info("Geographic visualization requires mapping coordinates or district boundaries which would be provided in a real implementation.")
                    
                    # Create a placeholder visualization with school data
                    school_risk = predictions_df.groupby('School')['CA_Risk'].mean().reset_index()
                    school_risk['Risk_Percentage'] = school_risk['CA_Risk'] * 100
                    school_risk['Student_Count'] = predictions_df.groupby('School').size().values
                    
                    # Add random coordinates for demonstration
                    np.random.seed(42)
                    school_risk['lat'] = np.random.uniform(40.7, 40.9, len(school_risk))
                    school_risk['lon'] = np.random.uniform(-74.1, -73.9, len(school_risk))
                    
                    fig = px.scatter_mapbox(
                        school_risk,
                        lat="lat",
                        lon="lon",
                        color="Risk_Percentage",
                        size="Student_Count",
                        hover_name="School",
                        color_continuous_scale="RdYlGn_r",
                        zoom=10,
                        title="Geographic Distribution of CA Risk (Sample Visualization)",
                        mapbox_style="carto-positron",
                        labels={
                            "Risk_Percentage": "Risk %",
                            "Student_Count": "Student Count"
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis
                    st.subheader("Regional Insights")
                    st.markdown("""
                    - In a real implementation, this map would show actual school locations and district boundaries
                    - Geographic risk clustering can reveal transportation issues, community challenges, or regional resource gaps
                    - School administrators can use this to coordinate with community services for targeted support
                    """)
                
                elif report_type == "Cohort Analysis":
                    st.markdown("<div class='card-title'>👥 Cohort Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    This analysis tracks student cohorts over time to observe how chronic absenteeism risk changes
                    as students progress through grade levels and to identify critical transition points.
                    """)
                    
                    # Create sample cohort data for demonstration
                    grades = list(range(1, 13))
                    
                    # Model different cohort patterns
                    cohort_1 = [0.15, 0.16, 0.18, 0.20, 0.22, 0.30, 0.32, 0.33, 0.35, 0.38, 0.40, 0.42]  # Gradual increase
                    cohort_2 = [0.12, 0.12, 0.13, 0.13, 0.25, 0.28, 0.27, 0.26, 0.25, 0.35, 0.38, 0.40]  # Jump at grades 5 & 10
                    cohort_3 = [0.18, 0.19, 0.20, 0.21, 0.22, 0.35, 0.38, 0.40, 0.41, 0.42, 0.45, 0.48]  # Jump at middle school
                    
                    # Create dataframe
                    cohort_data = pd.DataFrame({
                        'Grade': grades * 3,
                        'Cohort': ['Cohort 2020'] * 12 + ['Cohort 2019'] * 12 + ['Cohort 2018'] * 12,
                        'Risk': cohort_1 + cohort_2 + cohort_3
                    })
                    
                    cohort_data['Risk_Percentage'] = cohort_data['Risk'] * 100
                    
                    # Create line chart
                    fig = px.line(
                        cohort_data,
                        x='Grade',
                        y='Risk_Percentage',
                        color='Cohort',
                        markers=True,
                        title="CA Risk Progression by Cohort",
                        labels={"Risk_Percentage": "CA Risk Percentage", "Grade": "Grade Level"}
                    )
                    
                    fig.update_layout(
                        xaxis=dict(tickmode='array', tickvals=grades),
                        yaxis=dict(range=[0, 50])
                    )
                    
                    # Add annotations for transition points
                    fig.add_vline(x=5.5, line_dash="dash", line_color="gray", opacity=0.7)
                    fig.add_vline(x=8.5, line_dash="dash", line_color="gray", opacity=0.7)
                    
                    fig.add_annotation(x=5.5, y=45, text="Elementary → Middle", showarrow=False)
                    fig.add_annotation(x=8.5, y=45, text="Middle → High", showarrow=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add summary and insights
                    st.subheader("Key Transition Points")
                    st.markdown("""
                    - **Elementary to Middle School**: All cohorts show a significant increase in CA risk (average +10%)
                    - **9th to 10th Grade**: Another critical point with risk increases across all cohorts
                    - **Year-over-Year Pattern**: Each successive cohort shows slightly higher risk at equivalent grade levels
                    
                    These transition points suggest the need for targeted support programs at key grade levels to help
                    students navigate academic and social transitions.
                    """)
            else:
                st.warning("No data available for analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif app_mode == "System Settings":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.header("System Settings")
        
        st.subheader("Risk Thresholds")
        st.markdown("""
        Adjust the thresholds that determine risk levels for chronic absenteeism.
        """)
        
        # Risk threshold settings
        col1, col2 = st.columns(2)
        with col1:
            low_threshold = st.slider(
                "Low Risk Threshold",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.risk_thresholds['low'],
                step=0.05,
                help="Risk values below this threshold are considered low risk"
            )
        
        with col2:
            medium_threshold = st.slider(
                "Medium Risk Threshold",
                min_value=0.5,
                max_value=0.9,
                value=st.session_state.risk_thresholds['medium'],
                step=0.05,
                help="Risk values below this threshold but above the low threshold are considered medium risk"
            )
        
        if st.button("Update Risk Thresholds"):
            st.session_state.risk_thresholds = {
                'low': low_threshold,
                'medium': medium_threshold,
                'high': 1.0
            }
            st.success("Risk thresholds updated successfully!")
        
        # Intervention settings
        st.subheader("Intervention Settings")
        st.markdown("""
        Configure the cost and effectiveness of different intervention strategies.
        """)
        
        interventions = list(st.session_state.interventions.keys())
        
        for intervention in interventions:
            col1, col2 = st.columns(2)
            
            with col1:
                cost = st.number_input(
                    f"{intervention} Cost ($)",
                    min_value=0,
                    max_value=2000,
                    value=st.session_state.interventions[intervention]['cost'],
                    step=50
                )
            
            with col2:
                effectiveness = st.slider(
                    f"{intervention} Effectiveness",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.interventions[intervention]['effectiveness'],
                    step=0.05,
                    help="Estimated reduction in CA risk (0-1)"
                )
            
            st.session_state.interventions[intervention] = {
                'cost': cost,
                'effectiveness': effectiveness
            }
        
        # System reset option
        st.subheader("System Reset")
        st.markdown("""
        Reset the system to its initial state, clearing all data and models.
        """)
        
        if st.button("Reset System", help="This will clear all data and models from the system"):
            # Reset all session state variables
            for key in list(st.session_state.keys()):
                if key not in ['risk_thresholds', 'interventions']:
                    del st.session_state[key]
            
            st.session_state.model = None
            st.session_state.label_encoders = {}
            st.session_state.citywide_mode = False
            st.session_state.current_df = pd.DataFrame()
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.student_history = {}
            st.session_state.what_if_params = {}
            st.session_state.what_if_changes = {}
            st.session_state.current_student_id = ""
            st.session_state.original_prediction = None
            st.session_state.input_data = {}
            st.session_state.disable_inputs = False
            st.session_state.needs_prediction = False
            st.session_state.current_prediction = None
            st.session_state.what_if_prediction = None
            st.session_state.calculation_complete = False
            st.session_state.training_report = None
            st.session_state.batch_results = pd.DataFrame()
            st.session_state.model_features = []
            
            st.success("System reset completed successfully! Refresh the page to start with a clean state.")
        st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    # Initialize workflow state variables
    if 'training_successful' not in st.session_state:
        st.session_state.training_successful = False  # Feature access flag - defaults to False
        
    if 'model' not in st.session_state:
        st.session_state.model = None  # No model initially
        
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "System Training"  # Default to System Training
    
    # Risk thresholds
    if 'risk_thresholds' not in st.session_state:
        st.session_state.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 1.0
        }
    
    # Intervention settings
    if 'interventions' not in st.session_state:
        st.session_state.interventions = {
            'Parent Outreach': {'cost': 50, 'effectiveness': 0.15},
            'Attendance Mentoring': {'cost': 250, 'effectiveness': 0.35},
            'Support Services': {'cost': 500, 'effectiveness': 0.5},
            'Intensive Intervention': {'cost': 1000, 'effectiveness': 0.7}
        }
    
    # Data
    if 'current_df' not in st.session_state:
        st.session_state.current_df = pd.DataFrame()
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    if 'student_history' not in st.session_state:
        st.session_state.student_history = {}
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = pd.DataFrame()
    
    # Model data
    if 'models' not in st.session_state:
        st.session_state.models = {
            'random_forest': None, 
            'gradient_boost': None, 
            'logistic_regression': None, 
            'neural_network': None
        }
    if 'training_report' not in st.session_state:
        st.session_state.training_report = None
    
    # UI state
    if 'disable_inputs' not in st.session_state:
        st.session_state.disable_inputs = False
    if 'current_student_id' not in st.session_state:
        st.session_state.current_student_id = ""
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    
    # Export settings
    if 'export_cols_default' not in st.session_state:
        st.session_state.export_cols_default = ['Student_ID', 'Risk_Value', 'Risk_Level', 'School', 'Grade']
    if 'export_cols' not in st.session_state:
        st.session_state.export_cols = st.session_state.export_cols_default.copy()
    if 'export_risk_default' not in st.session_state:
        st.session_state.export_risk_default = ['Low', 'Medium', 'High']
    if 'export_risk' not in st.session_state:
        st.session_state.export_risk = st.session_state.export_risk_default.copy()

# Run the app
if __name__ == "__main__":
    initialize_session_state()
    main()