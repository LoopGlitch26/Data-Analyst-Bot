import streamlit as st
import pandas as pd
import ydata_profiling as yp
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import plotly.express as px

# Set up Streamlit app
st.title("Data Analysis and ML App")

# Initialize AI components
secret_key = st.secrets["MY_SECRET_KEY"]
llm = OpenAI(api_token=secret_key)

import os
from pathlib import Path

# Create a custom cache directory within the app's working directory
custom_cache_dir = Path(__file__).parent / "cache"
if not custom_cache_dir.exists():
    custom_cache_dir.mkdir()

# Initialize PandasAI with the custom cache directory
pandas_ai = PandasAI(llm, cache_dir=str(custom_cache_dir))

# Initialize variables
data_uploaded = False
eda_done = False
chat_done = False
modelling_done = False
df = None

# Page 1: Upload Dataset
st.header("Step 1: Upload Dataset")
st.info("Please upload your dataset in CSV format.")
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success("Dataset uploaded successfully!")
    data_uploaded = True

if data_uploaded:
    # Page 2: Exploratory Data Analysis (EDA)
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    st.write("Explore your dataset and generate a data report.")

    # Perform EDA using ydata_profiling
    profile_df = yp.ProfileReport(df)
    st_profile_report(profile_df)
    eda_done = True

if eda_done:
    # Page 3: Chat about Your Data
    st.header("Step 3: Chat about Your Data")
    st.write("Engage in a conversation with your data using AI.")

    # Chat about data using pandasai
    prompt = st.text_area("Enter your prompt")
    if st.button("Chat"):
        if prompt:
            with st.spinner("Generating response:"):
                st.write(pandas_ai.run(df, prompt=prompt))
            chat_done = True
        else:
            st.warning("Enter a prompt")

if chat_done:
    # Page 4: Machine Learning Modelling
    st.header("Step 4: Machine Learning Modelling")
    st.write("Select a target column and run machine learning models.")

    # Prepare data for modeling
    chosen_target = st.selectbox('Choose Target Column', df.columns)
    for col in df.columns:
        if df[col].nunique() == 2:
            df[col] = df[col].astype('category').cat.codes

    # Run machine learning models using pycaret
    if st.button('Run Modelling'):
        try:
            setup_env = setup(df, target=chosen_target, numeric_imputation='median', feature_selection=True)
        except Exception as e:
            st.error(f"An error occurred during setup: {e}")

        with st.spinner('Testing Models..'):
            setup_df = pull()
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
            modelling_done = True

if modelling_done:
    # Page 5: Download Analysis
    st.header("Step 5: Download Analysis")
    st.write("Download the analysis and best model.")

    # Download best model
    if st.button('Download Best Model'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

    # Optionally, you can add more download options for analysis reports

# Add a footer to the app
st.text("Powered by Streamlit & PandasAI")
