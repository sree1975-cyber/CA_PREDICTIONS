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
