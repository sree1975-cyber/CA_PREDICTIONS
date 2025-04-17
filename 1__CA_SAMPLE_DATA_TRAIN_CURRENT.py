import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Configure the app
st.set_page_config(
    page_title="Absenteeism Data Generator Pro",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Advanced Absenteeism Data Generator")
st.markdown("Generate customizable historical training data and current year data")

# Initialize session state
if 'days_completed' not in st.session_state:
    st.session_state.days_completed = 90
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

# Sidebar controls - Organized into expandable sections
with st.sidebar:
    st.header("⚙️ Data Parameters")
    
    with st.expander("📌 Core Settings", expanded=True):
        num_students = st.slider("Number of students", 100, 5000, 1000,
                                help="Total students to generate in the dataset")
        st.session_state.days_completed = st.slider(
            "Days completed in current year", 30, 180, st.session_state.days_completed,
            help="Simulates how far into the school year we are"
        )
    
    with st.expander("👥 Demographics"):
        male_ratio = st.slider("Male ratio (%)", 30, 70, 48) / 100
        female_ratio = st.slider("Female ratio (%)", 30, 70, 48) / 100
        other_ratio = 1 - male_ratio - female_ratio
    
    with st.expander("🍽 Socioeconomics"):
        free_meal = st.slider("Free meal (%)", 10, 60, 40) / 100
        reduced_meal = st.slider("Reduced meal (%)", 10, 50, 30) / 100
        paid_meal = 1 - free_meal - reduced_meal
    
    # NEW: Historical data specific controls
    with st.expander("🕰 Historical Data Settings"):
        base_attendance = st.slider("Base attendance rate (%)", 70, 95, 85,
                                  help="Average attendance percentage in historical data")
        year_variation = st.slider("Year-to-year variation (%)", 1, 20, 5,
                                 help="How much attendance varies between years")
        ca_threshold = st.slider("CA risk threshold (%)", 60, 90, 70,
                               help="Percentile cutoff for Chronic Absenteeism classification")

# Generate base student profiles (same as before)
def generate_student_profiles(num_students):
    current_year = datetime.now().year
    return pd.DataFrame({
        'Student_ID': [f"STU-{current_year}-{i:04d}" for i in range(1, num_students+1)],
        'School': np.random.choice([f"PS-{100+i}" for i in range(5)], num_students),
        'Grade': np.random.randint(1, 13, num_students),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_students, p=[male_ratio, female_ratio, other_ratio]),
        'Meal_Code': np.random.choice(['Free', 'Reduced', 'Paid'], num_students, p=[free_meal, reduced_meal, paid_meal]),
        'Academic_Performance': np.clip(np.random.normal(70, 15, num_students), 0, 100).astype(int),
    })

# Historical data generation with new parameters
def generate_training_data(base_df):
    years = [datetime.now().year - i for i in range(1, 4)]  # Last 3 years
    
    dfs = []
    for year in years:
        df = base_df.copy()
        df['Year'] = year
        
        # Use the base attendance rate with configured variation
        attendance_mean = base_attendance * 1.8  # Convert % to days (out of 180)
        variation = (year_variation/100) * attendance_mean
        attendance_base = np.clip(np.random.normal(attendance_mean, variation, len(df)), 0, 180)
        
        df['Present_Days'] = attendance_base.astype(int)
        df['Absent_Days'] = (180 - df['Present_Days']).clip(0, 180)
        df['Attendance_Percentage'] = (df['Present_Days'] / 180) * 100
        
        # Enhanced CA risk calculation with adjustable threshold
        risk_factors = (
            0.4 * (df['Attendance_Percentage'] < 85) +
            0.3 * (df['Meal_Code'] == 'Free') +
            0.2 * (df['Academic_Performance'] < 60) +
            0.1 * np.isin(df['Grade'], [6,7,8])
        )
        threshold = np.percentile(risk_factors, ca_threshold)
        df['CA_Status'] = np.where(risk_factors > threshold, 'CA', 'NO_CA')
        
        dfs.append(df)
    
    return pd.concat(dfs).sample(frac=1).reset_index(drop=True)

# Current year data generation
def generate_current_data(base_df, days_completed):
    df = base_df.copy()
    df['Year'] = datetime.now().year
    
    attendance_rate = np.clip(np.random.normal(base_attendance/100, 0.1, len(df)), 0, 1)
    df['Present_Days'] = (days_completed * attendance_rate).astype(int)
    df['Absent_Days'] = days_completed - df['Present_Days']
    df['Attendance_Percentage'] = (df['Present_Days'] / days_completed) * 100
    
    if 'CA_Status' in df.columns:
        df = df.drop(columns=['CA_Status'])
    
    return df

# Generate data button - now regenerates whenever clicked
if st.button("🔄 Generate/Update All Data", type="primary"):
    with st.spinner("Generating datasets..."):
        base_profiles = generate_student_profiles(num_students)
        st.session_state.train_df = generate_training_data(base_profiles)
        st.session_state.current_df = generate_current_data(base_profiles, st.session_state.days_completed)

# Display data if available
if st.session_state.train_df is not None and st.session_state.current_df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📜 Historical Training Data")
        st.dataframe(st.session_state.train_df.head())
        
        st.metric("Total Records", len(st.session_state.train_df))
        st.metric("CA Rate", f"{len(st.session_state.train_df[st.session_state.train_df['CA_Status'] == 'CA'])/len(st.session_state.train_df):.1%}")
        
        # Download training data
        train_csv = st.session_state.train_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Training Data (CSV)", 
            train_csv, 
            f"absenteeism_training_data_{len(st.session_state.train_df)}_records.csv",
            "text/csv",
            key="train_download"
        )
        
        fig1 = px.histogram(
            st.session_state.train_df, 
            x='Attendance_Percentage', 
            color='CA_Status',
            facet_col='Year',
            title='Historical Attendance by CA Status'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📆 Current Year Data")
        st.caption(f"Simulating {st.session_state.days_completed} days into the school year")
        st.dataframe(st.session_state.current_df.head())
        
        st.metric("Students", len(st.session_state.current_df))
        st.metric("Avg Attendance", f"{st.session_state.current_df['Attendance_Percentage'].mean():.1f}%")
        
        # Download current data
        current_csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Current Year Data (CSV)", 
            current_csv, 
            f"absenteeism_current_data_{datetime.now().year}.csv",
            "text/csv",
            key="current_download"
        )
        
        fig2 = px.scatter(
            st.session_state.current_df,
            x='Academic_Performance',
            y='Attendance_Percentage',
            color='Meal_Code',
            title=f'Current Year: {st.session_state.days_completed} Days Completed'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Clear data button at bottom
    st.markdown("---")
    if st.button("🧹 Clear All Data", type="secondary"):
        st.session_state.train_df = None
        st.session_state.current_df = None
        st.rerun()

# Instructions
with st.expander("ℹ️ How to use these datasets"):
    st.markdown("""
    **Training Data** (Historical):
    - Contains 3 years of complete data (180 days each)
    - Includes `CA_Status` for model training
    - Use in your "System Training" section
    
    **Current Year Data**:
    - Partial year data (adjust days completed)
    - No `CA_Status` (this is what you'll predict)
    - Use in your "Batch Prediction" section
    
    Both datasets maintain consistent student profiles across years.
    """)
