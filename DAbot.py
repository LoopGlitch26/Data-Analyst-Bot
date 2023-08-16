import streamlit as st
import pandas as pd
import ydata_profiling as yp

# Set up Streamlit app
st.title("Data Analyst Bot")

# Initialize variables
data_uploaded = False
eda_done = False
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
    
    # Display EDA report
    st.write(profile_df.to_html(), unsafe_allow_html=True)
    eda_done = True

# Add a footer to the app
st.text("Bravish @ 2023")
