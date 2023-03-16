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


text1 = st.text_input("Event name", value="")

text2 = st.text_input("Location", value="")

text3 = st.text_input("Venue cost", value="")

text4 = st.text_input("Number of Facilitators", value="")

text5 = st.text_input("Equipments Cost", value="")

text6 = st.text_input("Food and Beverage Cost", value="")

text7 = st.text_input("Accommodations Cost", value="")

text8 = st.text_input("Marketing and Advertising Cost", value="")

text9 = st.text_input("Duration(days)", value="")

text10 = st.text_input("Transportation&Communication Cost", value="")

text11 = st.text_input("Guest of honor Cost", value="")

text12 = st.text_input("Insurance Cost", value="")

text13 = st.text_input("Miscellaneous Expenses", value="")

if st.button("Process"):
        # Create a dictionary to store the input values
        data = {"Event name": [text1],
                "Location": [text2],
                "Venue cost": [text3],
                "Number of Facilitators": [text4],
                "Equipments Cost": [text5],
                "Food and Beverage Cost": [text6],
                "Accommodations Cost": [text7],
                "Marketing and Advertising Cost": [text8],
                "Duration(days)": [text9],
                "Transportation&Communication Cost": [text10],
                "Guest of honor Cost": [text11],
                "Insurance Cost": [text12],
                "Miscellaneous Expenses": [text13]}

        # Convert the dictionary to Pandas dataframe
        df = pd.DataFrame(data)
        df.shape
#         loaded_model = tf.keras.models.load_model('Twende/assets')

#         # make predictions on new data
#         predict = loaded_model.predict(df)
        
#         col1, col2 = st.beta_columns(2)
#         with col1:
#              st.write("Original values")
#              st.write(df)

#         with col2:
#             st.write("Predicted values")
#             st.write(predict)
    
