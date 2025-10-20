import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Linear Model Explorer")
st.header("for numeric datasets")
uploaded_file = st.file_uploader("Upload CSV or XLSX file here", type=["csv", "xlsx"])
if uploaded_file is not None:
    @st.cache_data
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    df = load_data(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    df = df.apply(pd.to_numeric, downcast='float', errors='coerce')
    numeric_columns = df.select_dtypes(include=float).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("The dataset must contain at least two numeric columns for modeling.")
    else:
        with st.form("Select Features for Linear Model"):
            target_variable = st.selectbox("Select Target Variable (Y)", options=numeric_columns)
            feature_variables = st.multiselect("Select Feature Variables (X)", options=[col for col in numeric_columns if col != target_variable])
            submitted = st.form_submit_button("Build Linear Model")
        if submitted:
            if not feature_variables:
                st.error("Please select at least one feature variable.")
            else:
                df_calc = df.dropna(subset=[target_variable] + feature_variables)
                X = df_calc[feature_variables]
                y = df_calc[target_variable]
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                st.subheader("Model Coefficients")
                coeff_df = pd.DataFrame(model.coef_, index=feature_variables, columns=["Coefficient"])
                st.dataframe(coeff_df)
                
                st.subheader("Model Performance")
                r_squared = model.score(X, y)
                st.write(f"R-squared: {r_squared:.4f}")
                
                st.subheader("Actual vs Predicted Plot")
                plt.figure(figsize=(10, 6))
                plt.scatter(y, y_pred, alpha=0.7)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted")
                st.pyplot(plt)



