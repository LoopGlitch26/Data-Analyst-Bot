import streamlit as st
import pandas as pd
import ydata_profiling as yp
from streamlit_pandas_profiling import st_profile_report

st.title("Data Analyst Bot")

data_uploaded = False
df = None

st.header("Please upload your dataset")
file = st.file_uploader("Upload CSV File", type=["csv"])
if file: 
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
    st.dataframe(df)
    data_uploaded = True

if data_uploaded:
    st.header("Exploratory Data Analysis (EDA)")

    # Perform EDA using ydata_profiling
    profile_df = yp.ProfileReport(df)
    st_profile_report(profile_df)
