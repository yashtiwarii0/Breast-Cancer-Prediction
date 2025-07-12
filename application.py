import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Breast Cancer Prediction App</h1>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('randomforest.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'randomforest.pkl' is in the project directory.")
        return None

# Load scaler function
@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.warning("Scaler file not found. Using StandardScaler with default parameters.")
        return StandardScaler()

# Feature information
feature_info = {
    'radius_mean': 'Mean of distances from center to points on the perimeter',
    'texture_mean': 'Standard deviation of gray-scale values',
    'smoothness_mean': 'Mean of local variation in radius lengths',
    'compactness_mean': 'Mean of perimeter² / area - 1.0',
    'symmetry_mean': 'Mean of symmetry',
    'fractal_dimension_mean': 'Mean of "coastline approximation" - 1',
    'radius_se': 'Standard error of radius',
    'texture_se': 'Standard error of texture',
    'smoothness_se': 'Standard error of smoothness',
    'compactness_se': 'Standard error of compactness',
    'concavity_se': 'Standard error of concavity',
    'concave_points_se': 'Standard error of concave points',
    'symmetry_se': 'Standard error of symmetry',
    'fractal_dimension_se': 'Standard error of fractal dimension',
    'smoothness_worst': 'Worst (largest) value of smoothness',
    'symmetry_worst': 'Worst (largest) value of symmetry',
    'fractal_dimension_worst': 'Worst (largest) value of fractal dimension'
}

# Sidebar for input
st.sidebar.markdown('<h2 class="sub-header">📊 Input Features</h2>', unsafe_allow_html=True)

# Create input fields
input_data = {}

# Mean features
st.sidebar.markdown("### Mean Features")
col1, col2 = st.sidebar.columns(2)

with col1:
    input_data['radius_mean'] = st.number_input('Radius Mean', min_value=0.0, max_value=50.0, value=14.0, step=0.1)
    input_data['texture_mean'] = st.number_input('Texture Mean', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
    input_data['smoothness_mean'] = st.number_input('Smoothness Mean', min_value=0.0, max_value=1.0, value=0.1, step=0.001)

with col2:
    input_data['compactness_mean'] = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, value=0.1, step=0.001)
    input_data['symmetry_mean'] = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, value=0.2, step=0.001)
    input_data['fractal_dimension_mean'] = st.number_input('Fractal Dimension Mean', min_value=0.0, max_value=1.0, value=0.06, step=0.001)

# Standard Error features
st.sidebar.markdown("### Standard Error Features")
col3, col4 = st.sidebar.columns(2)

with col3:
    input_data['radius_se'] = st.number_input('Radius SE', min_value=0.0, max_value=10.0, value=0.5, step=0.01)
    input_data['texture_se'] = st.number_input('Texture SE', min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    input_data['smoothness_se'] = st.number_input('Smoothness SE', min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    input_data['compactness_se'] = st.number_input('Compactness SE', min_value=0.0, max_value=1.0, value=0.03, step=0.001)

with col4:
    input_data['concavity_se'] = st.number_input('Concavity SE', min_value=0.0, max_value=1.0, value=0.03, step=0.001)
    input_data['concave_points_se'] = st.number_input('Concave Points SE', min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    input_data['symmetry_se'] = st.number_input('Symmetry SE', min_value=0.0, max_value=1.0, value=0.02, step=0.001)
    input_data['fractal_dimension_se'] = st.number_input('Fractal Dimension SE', min_value=0.0, max_value=1.0, value=0.003, step=0.001)

# Worst features
st.sidebar.markdown("### Worst Features")
col5, col6 = st.sidebar.columns(2)

with col5:
    input_data['smoothness_worst'] = st.number_input('Smoothness Worst', min_value=0.0, max_value=1.0, value=0.15, step=0.001)
    input_data['symmetry_worst'] = st.number_input('Symmetry Worst', min_value=0.0, max_value=1.0, value=0.3, step=0.001)

with col6:
    input_data['fractal_dimension_worst'] = st.number_input('Fractal Dimension Worst', min_value=0.0, max_value=1.0, value=0.08, step=0.001)

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">🔍 Prediction Results</h2>', unsafe_allow_html=True)
    
    if model is not None:
        # Make prediction
        if st.button('🔬 Predict', type='primary', use_container_width=True):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Scale the input data
            try:
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                # Display results
                if prediction == 0:  # Benign
                    st.markdown(f'''
                    <div class="prediction-box benign">
                        ✅ BENIGN<br>
                        Probability: {probability[0]:.2%}
                    </div>
                    ''', unsafe_allow_html=True)
                    st.success("The tumor is predicted to be benign (non-cancerous).")
                else:  # Malignant
                    st.markdown(f'''
                    <div class="prediction-box malignant">
                        ⚠️ MALIGNANT<br>
                        Probability: {probability[1]:.2%}
                    </div>
                    ''', unsafe_allow_html=True)
                    st.error("The tumor is predicted to be malignant (cancerous). Please consult with a medical professional.")
                
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Benign', 'Malignant'],
                        y=[probability[0], probability[1]],
                        marker_color=['#28a745', '#dc3545']
                    )
                ])
                fig.update_layout(
                    title='Prediction Probabilities',
                    yaxis_title='Probability',
                    xaxis_title='Class',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Please check your model file.")

with col2:
    st.markdown('<h2 class="sub-header">📋 Feature Information</h2>', unsafe_allow_html=True)
    
    # Feature information expander
    with st.expander("📖 Feature Descriptions", expanded=True):
        for feature, description in feature_info.items():
            st.write(f"**{feature}**: {description}")
    
    # Input summary
    st.markdown('<h3 class="sub-header">📊 Input Summary</h3>', unsafe_allow_html=True)
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df.T, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is a predictive model for educational purposes only. 
    Always consult with qualified healthcare professionals for medical advice and diagnosis.</p>
    <p>🔬 Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)

# About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### About Breast Cancer Prediction
    
    This application uses machine learning to predict whether a breast tumor is benign or malignant based on various features extracted from cell nuclei present in breast mass images.
    
    **Features used:**
    - **Mean features**: Average values of cell nuclei characteristics
    - **Standard Error features**: Standard error of cell nuclei characteristics  
    - **Worst features**: Largest (worst) values of cell nuclei characteristics
    
    **Model Performance:**
    - The Random Forest classifier is trained on the Wisconsin Breast Cancer Dataset
    - Features are standardized using StandardScaler for optimal performance
    
    **How to use:**
    1. Enter the feature values in the sidebar
    2. Click the "Predict" button to get the prediction
    3. View the probability scores and interpretation
    
    **Note:** This tool is for educational purposes only and should not replace professional medical diagnosis.
    """)

# Data visualization section
st.markdown('<h2 class="sub-header">📈 Feature Visualization</h2>', unsafe_allow_html=True)

# Create radar chart for current input
if st.checkbox("Show Feature Radar Chart"):
    categories = list(input_data.keys())
    values = list(input_data.values())
    
    # Normalize values for better visualization
    normalized_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Current Input'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Feature Values (Normalized)"
    )
    
    st.plotly_chart(fig, use_container_width=True)