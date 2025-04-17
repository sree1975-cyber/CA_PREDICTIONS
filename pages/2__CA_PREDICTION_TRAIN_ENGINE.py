import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import tempfile
import base64

# Configure the app
st.set_page_config(
    page_title="Chronic Absenteeism Predictor",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'citywide_mode' not in st.session_state:
    st.session_state.citywide_mode = False

# Title and description
st.title("🏫 Chronic Absenteeism Early Warning System")
st.markdown("Predict students at risk of chronic absenteeism (CA) using historical patterns and current data.")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", 
                          ["System Training", 
                           "Batch Prediction", 
                           "Single Student Check",
                           "Advanced Analytics"])

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
    """Train XGBoost model on the provided data"""
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
            return None, None
        
        # Split data
        X = df_processed.drop(['CA_Status', 'Student_ID'], axis=1, errors='ignore')
        y = df_processed['CA_Status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model with explicit binary classification parameters
        model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',  # Explicitly set binary classification
            random_state=42,
            eval_metric='logloss'  # Appropriate metric for binary classification
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, report
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None

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
        if isinstance(model, XGBClassifier):
            risk = model.predict_proba(df_processed)[:, 1]
        else:
            risk = model.predict(df_processed)
        
        return risk
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# System Training Section
if app_mode == "System Training":
    st.header("🔧 System Training")
    st.markdown("Upload historical data to train the prediction model.")
    
    with st.expander("📋 Data Requirements", expanded=True):
        st.markdown("""
        Your Excel file should include these columns:
        - **Student_ID**: Unique identifier (optional)
        - **School**: School name/code
        - **Grade**: Grade level (1-12)
        - **Gender**: Male/Female/Other
        - **Present_Days**: Number of days present
        - **Absent_Days**: Number of days absent
        - **Meal_Code**: Free/Reduced/Paid (SES proxy)
        - **Academic_Performance**: Score (0-100)
        - **CA_Status**: Chronic Absenteeism status (YES/NO or 1/0)
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
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Train model
                if st.button("Train Prediction Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes"):
                        model, report = train_model(df)
                        
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
                                'Feature': model.feature_names_in_,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feature_importance.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Most Important Features'
                            )
                            st.plotly_chart(fig)
                            
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
                                    bins=[0, 0.3, 0.7, 1],
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
        with st.form("single_student_form"):
            st.subheader("Student Information")
            
            col1, col2 = st.columns(2)
            with col1:
                student_id = st.text_input("Student ID (Optional)")
                grade = st.selectbox("Grade", range(1, 13), index=5)
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
                meal_code = st.selectbox("Meal Code", ["Free", "Reduced", "Paid", "Unknown"])
            
            with col2:
                present_days = st.number_input("Present Days", min_value=0, max_value=365, value=45)
                absent_days = st.number_input("Absent Days", min_value=0, max_value=365, value=10)
                academic_performance = st.number_input("Academic Performance (0-100)", 
                                                    min_value=0, max_value=100, value=75)
                
                # Transfer student options
                if st.session_state.citywide_mode:
                    transferred = st.checkbox("Transferred student?")
                    if transferred:
                        prev_ca = st.selectbox("Previous school CA status", 
                                             ["Unknown", "Yes", "No"])
            
            submitted = st.form_submit_button("Check Risk", type="primary")
            
            if submitted:
                # Prepare input data
                input_data = {
                    'Student_ID': student_id,
                    'Grade': grade,
                    'Gender': gender,
                    'Present_Days': present_days,
                    'Absent_Days': absent_days,
                    'Meal_Code': meal_code,
                    'Academic_Performance': academic_performance
                }
                
                # Calculate attendance
                attendance_pct = (present_days / (present_days + absent_days)) * 100
                
                # Predict
                risk = predict_ca_risk(input_data, st.session_state.model)
                
                if risk is not None:
                    risk = float(risk[0])
                    
                    # Adjust for transfer history if applicable
                    if st.session_state.citywide_mode and transferred and prev_ca == "Yes":
                        risk = min(risk * 1.4, 0.99)  # Increase risk by 40%
                    
                    # Determine risk level
                    if risk < 0.3:
                        risk_level = "Low"
                        risk_class = "risk-low"
                    elif risk < 0.7:
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
                    
                    # Risk gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk * 100,
                        number={'suffix': "%"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "CA Risk Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk * 100
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key factors
                    st.subheader("Key Risk Factors")
                    
                    # Simulate feature importance (would use SHAP values in production)
                    factors = {
                        'Attendance <90%': max(0, (0.9 - (attendance_pct/100))) * 0.4,
                        'Low Grades': max(0, (70 - academic_performance)/70) * 0.3,
                        'Free/Reduced Meals': 0.2 if meal_code in ['Free', 'Reduced'] else 0,
                        'Grade Level': (grade/12) * 0.1  # Higher grades often have higher CA
                    }
                    
                    # Normalize to sum to risk
                    factor_sum = sum(factors.values())
                    if factor_sum > 0:
                        factors = {k: (v/factor_sum)*risk for k, v in factors.items()}
                    
                    fig2 = px.bar(
                        x=list(factors.values()),
                        y=list(factors.keys()),
                        orientation='h',
                        title='Contributing Risk Factors',
                        labels={'x': 'Contribution to Risk', 'y': 'Factor'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
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
    else:
        # Load sample data or uploaded data
        if 'current_df' not in st.session_state or st.session_state.current_df.empty:
            st.warning("No prediction data available. Please run batch predictions first.")
            st.stop()
        
        df = st.session_state.current_df
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization",
            [
                "Risk Distribution by School",
                "Attendance vs. Academic Performance",
                "Risk Heatmap by Grade & SES",
                "Temporal Attendance Trends"
            ]
        )
        
        if viz_option == "Risk Distribution by School":
            st.subheader("Risk Distribution by School")
            
            if 'School' not in df.columns:
                st.warning("School data not available in predictions.")
            else:
                # Group by school
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
                # Decode Meal_Code if encoded
                try:
                    if 'Meal_Code' in st.session_state.label_encoders:
                        le = st.session_state.label_encoders['Meal_Code']
                        df['Meal_Code'] = le.inverse_transform(df['Meal_Code'])
                except:
                    pass
                
                # Pivot table for heatmap
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
            
            # Get current year
            current_year = datetime.now().year
            
            # Simulate temporal data
            dates = pd.date_range(
                start=datetime(current_year, 1, 1),
                end=datetime(current_year, 12, 31),
                freq='W'
            )
            
            # Simulate attendance trends
            trend_data = pd.DataFrame({
                'Date': dates,
                'Attendance': np.sin(np.linspace(0, 10, len(dates))) * 0.1 + 0.8,
                'CA_Risk': np.cos(np.linspace(0, 8, len(dates))) * 0.1 + 0.3
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['Attendance'],
                name='Attendance Rate',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['CA_Risk'],
                name='CA Risk',
                yaxis='y2',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Weekly Attendance and CA Risk Trends',
                yaxis=dict(title='Attendance Rate'),
                yaxis2=dict(
                    title='CA Risk',
                    overlaying='y',
                    side='right'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
