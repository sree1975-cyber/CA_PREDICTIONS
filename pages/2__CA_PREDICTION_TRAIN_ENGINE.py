#BLOCK 1: Imports and Initial Setup
# 1. Core functionality imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 2. Visualization imports
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 3. Machine learning imports
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 4. Configure app settings
st.set_page_config(
    page_title="Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)
#BLOCK 2: Custom Styling
# 1. Custom CSS for consistent styling
st.markdown("""
<style>
    /* Main container padding */
    .main {padding: 2rem;}
    
    /* Metric boxes styling */
    .stMetric {
        border: 1px solid #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Risk level color coding */
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-medium {color: #ffa500; font-weight: bold;}
    .risk-low {color: #2ecc71; font-weight: bold;}
    
    /* Plot containers */
    div.stPlotlyChart, div.stPyplot {
        border: 1px solid #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Form elements */
    .stSlider, .stSelectbox, .stNumberInput {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

#BLOCK 3: Session State Initialization
# Initialize all session state variables with defaults
if 'model' not in st.session_state:
    st.session_state.model = None
    
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
    
if 'current_df' not in st.session_state:
    st.session_state.current_df = pd.DataFrame()
    
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

# Risk thresholds configuration
if 'risk_thresholds' not in st.session_state:
    st.session_state.risk_thresholds = {
        'low': 0.3,
        'medium': 0.7, 
        'high': 1.0
    }
#BLOCK 4: Data Preprocessing Functions
def preprocess_data(df, is_training=True):
    """
    Prepares data for model training or prediction
    - Handles missing values
    - Encodes categorical variables
    - Calculates derived features
    """
    df = df.copy()
    
    # 1. Calculate attendance percentage if not present
    if 'Attendance_Percentage' not in df.columns:
        if all(col in df.columns for col in ['Present_Days', 'Absent_Days']):
            total_days = df['Present_Days'] + df['Absent_Days']
            df['Attendance_Percentage'] = (df['Present_Days'] / total_days) * 100
    
    # 2. Handle categorical encoding
    cat_cols = ['Gender', 'Meal_Code', 'School']
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                # Initialize and fit encoder during training
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                # Handle unknown categories during prediction
                if col in st.session_state.label_encoders:
                    # Replace unknown categories with most frequent
                    known_categories = st.session_state.label_encoders[col].classes_
                    df[col] = df[col].apply(lambda x: x if x in known_categories else known_categories[0])
                    df[col] = st.session_state.label_encoders[col].transform(df[col])
    
    return df

#BLOCK 5: Model Training Functions
def train_model(df):
    """
    Trains an ensemble model (XGBoost + RandomForest)
    Returns:
    - Trained model
    - Classification report
    - Feature importance data
    """
    try:
        # 1. Preprocess the data
        df_processed = preprocess_data(df)
        
        # 2. Handle target variable encoding
        if df_processed['CA_Status'].dtype == 'object':
            df_processed['CA_Status'] = df_processed['CA_Status'].map({'NO_CA': 0, 'CA': 1})
        df_processed['CA_Status'] = df_processed['CA_Status'].astype(int)
        
        # 3. Verify binary target
        if set(df_processed['CA_Status'].unique()) != {0, 1}:
            raise ValueError("Target must be binary (0/1)")
        
        # 4. Prepare features and target
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 6. Initialize models
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # 7. Create and train ensemble
        model = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf)],
            voting='soft'
        )
        model.fit(X_train, y_train)
        
        # 8. Evaluate model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 9. Get feature importance (using XGBoost)
        feature_importance = pd.DataFrame({
            'Feature': model.named_estimators_['xgb'].feature_names_in_,
            'Importance': model.named_estimators_['xgb'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return model, report, feature_importance
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None
    #BLOCK 6: Prediction Functions
 def predict_ca_risk(input_data, model):
    """
    Makes predictions using trained model
    Handles data preprocessing and feature alignment
    """
    try:
        # 1. Convert input to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # 2. Preprocess the data
        df_processed = preprocess_data(df, is_training=False)
        
        # 3. Align features with training data
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0  # Add missing columns with default value
            df_processed = df_processed[model.feature_names_in_]  # Reorder columns
        
        # 4. Make prediction
        if isinstance(model, (XGBClassifier, VotingClassifier)):
            risk = model.predict_proba(df_processed)[:, 1]  # Probability of class 1
        else:
            risk = model.predict(df_processed)
        
        return risk
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

#BLOCK 7: Visualization Functions
def plot_feature_importance(feature_importance):
    """
    Creates a bar plot of feature importance
    Uses Plotly for interactive visualization
    """
    if feature_importance is not None:
        st.subheader("Top Predictive Factors")
        
        # Limit to top 10 features for readability
        top_features = feature_importance.head(10)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            labels={'Importance': 'Relative Importance', 'Feature': ''},
            height=400  # Fixed height for consistency
        )
        
        # Improve layout
        fig.update_layout(
            margin=dict(l=100, r=50, t=50, b=50),
            xaxis_title="Importance Score",
            yaxis_title="",
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_risk_gauge(risk_score, thresholds):
    """
    Creates an interactive gauge chart for risk visualization
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
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
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,  # Fixed height
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

#BLOCK 8: System Training Section
def show_training_section():
    st.header("🔧 System Training")
    st.markdown("Upload historical data to train the prediction model.")
    
    # Data requirements expander
    with st.expander("📋 Data Requirements", expanded=True):
        st.markdown("""
        Required columns:
        - Student_ID, School, Grade, Gender
        - Present_Days, Absent_Days  
        - Meal_Code, Academic_Performance
        - CA_Status (YES/NO or 1/0)
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Historical Data", 
        type=["xlsx", "csv"]
    )
    
    if uploaded_file:
        try:
            # Read file based on type
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = {'Grade', 'Gender', 'Present_Days', 'Absent_Days', 
                           'Meal_Code', 'Academic_Performance', 'CA_Status'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Train model button
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training in progress..."):
                        model, report, feature_importance = train_model(df)
                        
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
                            
                            # Show feature importance
                            plot_feature_importance(feature_importance)
                            
                            # Download model option
                            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                                joblib.dump(model, tmp.name)
                                with open(tmp.name, 'rb') as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="ca_model.pkl">Download Model</a>'
                                    st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

#9BLOCK 9: Single Student Analysis
def show_single_student_section():
    st.header("👤 Single Student Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the System Training section.")
        return
    
    # Student input form
    with st.form(key='student_form'):
        st.subheader("Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("Student ID (Optional)")
            grade = st.selectbox("Grade", range(1, 13), index=5)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
            meal_code = st.selectbox("Meal Status", ["Free", "Reduced", "Paid", "Unknown"])
        
        with col2:
            present_days = st.number_input("Present Days", min_value=0, max_value=365, value=45)
            absent_days = st.number_input("Absent Days", min_value=0, max_value=365, value=10)
            academic_performance = st.number_input("Academic Performance (0-100)", 
                                                min_value=0, max_value=100, value=75)
        
        submitted = st.form_submit_button("Analyze Risk")
    
    # Process form submission
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
        
        # Calculate attendance percentage
        attendance_pct = (present_days / (present_days + absent_days)) * 100
        
        # Make prediction
        risk = predict_ca_risk(input_data, st.session_state.model)
        
        if risk is not None:
            risk = float(risk[0])
            
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
            st.subheader("Analysis Results")
            
            # Metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stMetric">
                    <h3>Risk Level</h3>
                    <p class="{risk_class}" style="font-size: 2rem;">{risk_level}</p>
                    <p>Probability: {risk:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stMetric">
                    <h3>Attendance</h3>
                    <p style="font-size: 2rem;">{attendance_pct:.1f}%</p>
                    <p>{present_days} present / {absent_days} absent</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            plot_risk_gauge(risk, thresholds)
            
            # Recommendations
            st.subheader("Recommended Actions")
            if risk_level == "High":
                st.markdown("""
                - Immediate counselor meeting
                - Parent/guardian notification
                - Attendance improvement plan
                """)
            elif risk_level == "Medium":
                st.markdown("""
                - Monthly check-ins
                - Mentor assignment
                - Academic support
                """)
            else:
                st.markdown("""
                - Regular monitoring
                - Positive reinforcement
                """)
#BLOCK 10: Main App Navigation
def main():
    st.title("🏫 Student Absenteeism Predictor")
    
    # Sidebar navigation
    app_mode = st.sidebar.radio(
        "Navigation",
        ["System Training", "Single Student Analysis"],
        index=0
    )
    
    # Show selected section
    if app_mode == "System Training":
        show_training_section()
    elif app_mode == "Single Student Analysis":
        show_single_student_section()

if __name__ == "__main__":
    main()
