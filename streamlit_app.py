import streamlit as st
import pandas as pd

# Title of the app
st.title("Twende an AI model for predicting cost of an event")

# Upload CSV file
csv_file = st.file_uploader("Upload CSV", type=["csv"])

# Check if a file has been uploaded
if csv_file is not None:
    # Load CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Show the DataFrame in the app
    st.write("Original DataFrame:")
    st.write(df)

    # Process the DataFrame (e.g. drop NaN values)
    df_processed = df.dropna()

    # Show the processed DataFrame in the app
    st.write("Processed DataFrame:")
    st.write(df_processed)
