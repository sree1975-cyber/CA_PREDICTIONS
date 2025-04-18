# 1. Import Libraries
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

# 2. Streamlit App Config and Custom CSS
st.set_page_config(
    page_title="Enhanced Chronic Absenteeism Predictor",
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
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .shap-watermark {display: none !important;}
    div.stPlotlyChart {border: 1px solid #f0f2f6; border-radius: 0.5rem;}
    div.stShap {width: 100% !important; margin: 0 auto !important;}
    div.stShap svg {width: 100% !important; height: auto !important;}
    .stSlider {padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# 3. Initialize Session State
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
    st.session_state.what_if_params = {
        'attendance': 0,
        'performance': 0
    }

# 4. Helper: Generate Sample Data (unchanged)
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

# 5. Helper: Preprocess Data (fixed for unknowns)
def preprocess_data(df, is_training=True):
    """
    Preprocess the input data for training or prediction.
    Handles unknown categories by mapping them to a special value.
    """
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
                # Add 'Unknown' as a class if not present
                unique_vals = list(df[col].unique())
                if 'Unknown' not in unique_vals:
                    unique_vals.append('Unknown')
                le.fit(unique_vals)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                df[col] = le.transform(df[col])
                st.session_state.label_encoders[col] = le
            else:
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    df[col] = le.transform(df[col])
    return df

# 6. Helper: Train Model (unchanged)
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

# 7. Helper: Predict CA Risk (unchanged)
def predict_ca_risk(input_data, model):
    """Predict CA risk for input data"""
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

# 8. Helper: Plot Feature Importance (Plotly, replaces SHAP summary)
def plot_feature_importance(model, features):
    """
    Plot feature importance using Plotly for better readability.
    """
    st.subheader("Feature Importance (XGBoost)")
    if hasattr(model, 'named_estimators_'):
        xgb_model = model.named_estimators_['xgb']
        importance = xgb_model.feature_importances_
        feature_names = xgb_model.feature_names_in_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig = px.bar(
            fi_df, 
            x='Importance', y='Feature', 
            orientation='h', 
            title='Feature Impact on Chronic Absenteeism Risk',
            height=600
        )
        fig.update_layout(
            margin=dict(l=150, r=50, t=60, b=50),
            xaxis_title="Importance",
            yaxis_title="Features",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

# 9. Helper: Plot SHAP for Single Student (horizontal bar, readable)
def plot_student_shap(explainer, shap_values, features):
    """
    Display SHAP values for a single student as a horizontal bar chart.
    """
    st.subheader("Risk Factor Breakdown (Top Features)")
    vals = shap_values[0]
    feature_names = features.columns
    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': vals})
    shap_df['abs_SHAP'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values('abs_SHAP', ascending=False).head(10)
    fig = px.bar(
        shap_df[::-1],  # reverse for descending order in plot
        x='SHAP Value', y='Feature',
        orientation='h',
        color='SHAP Value',
        color_continuous_scale='RdBu',
        title='Top 10 Feature Impacts for This Student',
        height=400
    )
    fig.update_layout(
        margin=dict(l=150, r=50, t=60, b=50),
        xaxis_title="SHAP Value (Impact on Risk)",
        yaxis_title="Feature",
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- The rest of your code remains unchanged except for the following two places: ---

# 10. In System Training Section (replace SHAP summary plot)
# Replace:
# plot_shap_summary(explainer, shap_values, X_train)
# With:
# plot_feature_importance(model, X_train)

# 11. In Single Student Check (replace SHAP force plot with horizontal bar)
# Replace the entire SHAP explanation block with:
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
        plot_student_shap(explainer, shap_values, df_processed)
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
