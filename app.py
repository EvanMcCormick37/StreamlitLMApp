import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Linear Model Explorer",
    page_icon="📊",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

SAMPLE_DATASETS = {
    "Manufacturing": "data/manufacturing.csv",
    "Real Estate": "data/realestate.csv"
}

@st.cache_data
def load_csv_or_excel(file):
    return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

@st.cache_data
def load_sample_data(dataset_name):
    return pd.read_csv(SAMPLE_DATASETS[dataset_name])

@st.cache_data
def prepare_numeric_data(df, threshold=0.5):
    numeric_cols = []
    for col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        if numeric_series.notna().sum() / len(df) >= threshold:
            numeric_cols.append(col)
    
    return df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0), numeric_cols

def build_and_display_model(df, features, target):
    X, y = df[features], df[target]
    
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    
    # Use columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r_squared:.4f}")
    with col2:
        st.metric("Features Used", len(features))
    with col3:
        st.metric("Samples", len(df))
    
    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Coefficients", "📈 Performance Plot", "📋 Predictions"])
    
    with tab1:
        st.subheader("Model Coefficients")
        coeff_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_,
            'Abs Coefficient': np.abs(model.coef_)
        }).sort_values('Abs Coefficient', ascending=False)
        
        st.dataframe(coeff_df, use_container_width=True, hide_index=True)
        
        # Bar chart of coefficients
        st.bar_chart(coeff_df.set_index('Feature')['Coefficient'],
                     x_label = "y-Coefficient",
                     y_label = "Feature",
                     horizontal = True)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Actual vs Predicted Values", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        predictions_df = pd.DataFrame({
            'Actual': y.values,
            'Predicted': y_pred,
            'Residual': y.values - y_pred
        })
        st.dataframe(predictions_df.head(20), use_container_width=True)
        st.download_button(
            "📥 Download Full Predictions",
            predictions_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

# Header with icon and description
st.title("Linear Model Data Explorer")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Upload your dataset or try one of our sample datasets to get started.</h4>
</div>
""", unsafe_allow_html=True)

# Sidebar for data input
with st.sidebar:
    st.header("Data Input")
    
    uploaded_file = st.file_uploader("Upload Your Data", type=["csv", "xlsx"])
    st.markdown("""
        <p>When uploading a dataset, keep in mind that only numeric columns will be considered for modeling.</p>
        <p>Missing values will be automatically <b>coerced to 0</b>.</p>
        """, unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Or Analyze a Sample Dataset")
    with st.form("sample_form"):
        sample = st.selectbox("Select Dataset", options=list(SAMPLE_DATASETS.keys()))
        if st.form_submit_button("Load Sample", use_container_width=True):
            st.session_state.df = load_sample_data(sample)
            st.success(f"Loaded {sample} dataset!")

    if 'df' in st.session_state:
        st.divider()
        st.info(f"Dataset loaded: {len(st.session_state.df)} rows")

# Load uploaded file
if uploaded_file:
    st.session_state.df = load_csv_or_excel(uploaded_file)

# Main content area
if 'df' in st.session_state:
    df = st.session_state.df
    
    # Use expander for data preview
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    df_numeric, numeric_cols = prepare_numeric_data(df)
    
    with st.expander("View Numeric Data", expanded=False):
        st.dataframe(df_numeric.head(10), use_container_width=True)
        st.caption(f"Showing {len(numeric_cols)} numeric columns")
    
    if len(numeric_cols) < 2:
        st.error("⚠️ Dataset must contain at least two numeric columns for modeling.")
    else:
        st.divider()
        st.header("Configure Your Model")
        
        # Use columns for form layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target = st.selectbox("🎯 Target Variable (Y)", options=numeric_cols)
        
        with col2:
            use_all = st.checkbox("Use All Features", value=False)
        
        if not use_all:
            features = st.multiselect(
                "📊 Feature Variables (X)", 
                options=[col for col in numeric_cols if col != target],
                help="Select one or more features for your model"
            )
        else:
            features = [col for col in numeric_cols if col != target]
            st.info(f"Using all {len(features)} available features")
    
        if not features:
            st.error("❌ Please select at least one feature variable.")
        else:
            with st.spinner("Building model..."):
                st.divider()
                st.header("Model Results")
                build_and_display_model(df_numeric, features, target)
else:
    # Empty state with better visuals
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Get Started</h2>
        <p style='font-size: 18px; color: #666;'>Upload a dataset or select a sample from the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)