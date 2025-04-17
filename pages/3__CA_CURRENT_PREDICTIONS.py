import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- App Config ---
st.set_page_config(layout="wide", page_title="CA Predictor Pro+", page_icon="🚨")
st.title("🚨 Chronic Absenteeism Risk Intelligence System")

# --- Custom CSS ---
st.markdown("""
<style>
.risk-card {
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.high-risk { background-color: #ffebee; border-left: 5px solid #f44336; }
.medium-risk { background-color: #fff8e1; border-left: 5px solid #ffc107; }
.low-risk { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
.feature-importance { background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# --- Safe Model Loader ---
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model(uploaded_model):
    try:
        model = joblib.load(uploaded_model)
        # Flexible attribute fallback
        if not hasattr(model, 'feature_names_in_'):
            if hasattr(model, 'get_booster'):
                model.feature_names_in_ = model.get_booster().feature_names
            else:
                model.feature_names_in_ = list(getattr(model, 'feature_names', []))
        if not hasattr(model, 'label_encoders_'):
            model.label_encoders_ = getattr(model, 'label_encoders', {})
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# --- Safe Preprocessing ---
def preprocess_data(df, model):
    df = df.copy()
    # Handle attendance percentage column name variations
    if 'Attendance_Percentage' in df.columns:
        df['Attendance_Pct'] = df['Attendance_Percentage']
    elif 'Attendance_Pct' not in df.columns:
        if 'Present_Days' in df.columns and 'Absent_Days' in df.columns:
            total_days = df['Present_Days'] + df['Absent_Days']
            df['Attendance_Pct'] = (df['Present_Days'] / total_days.replace(0, 1)) * 100

    # Encode categoricals if encoders exist
    label_encoders = getattr(model, 'label_encoders_', {})
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])

    # Feature alignment
    if hasattr(model, 'feature_names_in_'):
        missing = set(model.feature_names_in_) - set(df.columns)
        for col in missing:
            df[col] = 0
        df = df[model.feature_names_in_]

    # Type safety
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype('category').cat.codes

    return df.astype(np.float32)

# --- Risk Explanation ---
def explain_risk(student_data):
    reasons = []
    if student_data.get('Attendance_Pct', 100) < 85:
        severity = "critical" if student_data['Attendance_Pct'] < 75 else "high"
        reasons.append(f"📉 <b>{severity.title()} attendance risk</b> ({student_data['Attendance_Pct']:.1f}%)")
    if student_data.get('Academic_Performance', 100) < 60:
        reasons.append(f"📚 <b>Low academic performance</b> (score: {student_data['Academic_Performance']})")
    if student_data.get('Meal_Code') == 'Free':
        reasons.append("🏠 <b>Economic disadvantage</b> (qualifies for free meals)")
    if student_data.get('Grade') in [6,7,8]:
        reasons.append("👦 <b>Middle school risk factor</b> (grades 6-8 have higher CA rates)")
    return reasons if reasons else ["No major risk factors identified"]

# --- Intervention Recommendations ---
def get_interventions(risk_level, reasons):
    interventions = {
        'High': [
            "🚑 Immediate counselor meeting",
            "📞 Same-day parent notification",
            "📝 Attendance improvement contract",
            "👨🏫 Daily check-ins for 2 weeks"
        ],
        'Medium': [
            "📅 Weekly mentor meetings",
            "📱 Bi-weekly parent updates",
            "🎯 Targeted tutoring sessions",
            "📊 Monthly progress reviews"
        ],
        'Low': [
            "📢 Positive reinforcement",
            "👀 Quarterly monitoring",
            "🏫 Encourage extracurriculars",
            "📋 Annual review"
        ]
    }
    base_actions = interventions.get(risk_level, [])
    if any("attendance" in reason.lower() for reason in reasons):
        base_actions.append("⏰ Implement attendance tracking system")
    if any("academic" in reason.lower() for reason in reasons):
        base_actions.append("📖 Assign subject-specific tutor")
    return base_actions

# --- Main App ---
def main():
    # Model Upload
    with st.expander("🔧 STEP 1: UPLOAD TRAINED MODEL", expanded=True):
        model_file = st.file_uploader("Upload model.pkl", type="pkl")
    model = load_model(model_file) if model_file else None

    # Initialize session state for analysis results
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    if model:
        # Data Upload
        with st.expander("📁 STEP 2: UPLOAD CURRENT STUDENT DATA", expanded=True):
            data_file = st.file_uploader("Upload students.csv", type="csv")
        
        if data_file:
            try:
                df = pd.read_csv(data_file)
                
                if st.button("🚀 ANALYZE STUDENT RISKS", type="primary"):
                    with st.spinner("Processing data..."):
                        df_processed = preprocess_data(df, model)
                        # Predict
                        df['Risk_Score'] = model.predict_proba(df_processed)[:, 1]
                        df['Risk_Level'] = pd.cut(
                            df['Risk_Score'],
                            bins=[0, 0.3, 0.7, 1],
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        # Store results in session state
                        st.session_state.processed_df = df
                        st.session_state.analysis_complete = True
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Data processing failed: {str(e)}")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        st.success("Analysis Complete!")
        st.header("📊 RISK LANDSCAPE OVERVIEW")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df['Risk_Score'].mean()*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': df['Risk_Score'].mean()*100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'Attendance_Percentage' in df.columns:
                fig = px.scatter(
                    df,
                    x='Attendance_Percentage',
                    y='Academic_Performance',
                    color='Risk_Score',
                    size='Risk_Score',
                    hover_name='Student_ID' if 'Student_ID' in df.columns else None,
                    hover_data=['Grade', 'School', 'Meal_Code'] if all(x in df.columns for x in ['Grade', 'School', 'Meal_Code']) else None,
                    color_continuous_scale='reds',
                    title="Student Risk Positioning"
                )
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Attendance percentage data missing for scatter plot.")

        with col3:
            risk_counts = df['Risk_Level'].value_counts()
            fig = px.pie(
                risk_counts,
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.4,
                color=risk_counts.index,
                color_discrete_map={
                    'High': '#FF5252',
                    'Medium': '#FFC107',
                    'Low': '#4CAF50'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Student-level Intelligence ---
        st.header("🔍 STUDENT RISK PROFILES")
        if 'Student_ID' in df.columns:
            selected_id = st.selectbox(
                "Select Student ID for Detailed Analysis",
                df['Student_ID'].unique()
            )
            student_data = df[df['Student_ID'] == selected_id].iloc[0].to_dict()
            risk_class = student_data['Risk_Level'].lower() + "-risk"
            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <h3>Student: {selected_id}</h3>
                    <p><b>Risk Level</b>: {student_data['Risk_Level']} ({student_data['Risk_Score']:.0%})</p>
                    <p><b>Grade</b>: {student_data.get('Grade', 'N/A')} | <b>School</b>: {student_data.get('School', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("📋 Risk Factors")
                reasons = explain_risk(student_data)
                for reason in reasons:
                    st.markdown(f"<div class='feature-importance'>{reason}</div>", unsafe_allow_html=True)
            with col_right:
                st.subheader("🛡️ Recommended Interventions")
                interventions = get_interventions(student_data['Risk_Level'], reasons)
                for action in interventions:
                    st.markdown(f"- {action}")

        # --- Download Full Results ---
        st.download_button(
            "💾 DOWNLOAD COMPLETE ANALYSIS",
            df.to_csv(index=False),
            "ca_risk_analysis.csv",
            "text/csv"
        )

        st.markdown("---")  # Adds a horizontal line for visual separation
        
        # Add New Analysis button with some explanatory text
        st.markdown("### Start a new analysis")
        st.write("Clear current results to analyze new data")
        if st.button('🔄 NEW ANALYSIS', type="secondary"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
