import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

# Title of the app
st.title("Twende an AI model for predicting cost of an event")

# Upload CSV file
csv_file = st.file_uploader("Upload CSV", type=["csv"])
data_frame
# Check if a file has been uploaded
if csv_file is not None:
    # Load CSV file into a Pandas DataFrame
    
    data_frame= pd.read_csv(csv_file)

    # Show the DataFrame in the app
    st.write("Original DataFrame:")
 st.write(data_frame)
   
