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
                "Miscellaneous Expenses": [text13],
                'Event name_Dar es Salaam Startup': [0, 0],
                'Event name_Pitch Night Tanzania': [0, 0],
                'Event name_STEM Education and Innovation Conference': [0, 0],
                'Event name_STEM for Girls Workshop': [0, 0],
                'Event name_Tanzania Entrepreneurship Summit': [0, 0],
                'Event name_Tanzania Fashion Week': [0, 0],
                'Event name_Tanzania Food Festival': [0, 0],
                'Event name_Tanzania Science Fair': [0, 0],
                'Event name_Tanzania Tech Summit': [0, 0],
                'Event name_The Education Summit': [0, 0],
                'Event name_The Innovation Week': [0, 0],
                'Event name_Twende Build It': [0, 1],
                'Event name_Twende CCB': [0, 0],
                'Event name_Twende Cultural Week': [0, 0],
                'Event name_Twende Environmental Expo': [0, 0],
                'Event name_Twende Farmers week': [0, 0],
                'Event name_Twende Jamii Tech Incubation Program': [0, 0],
                'Event name_Twende STEM Outreach': [0, 0],
                'Event name_Twende kumasi': [1, 0],
                'Location_Arusha': [1, 0],
                'Location_Dar-es-salaam': [0, 0],
                'Location_Dodoma': [0, 0],
                'Location_Geita': [0, 0],
                'Location_Iringa': [0, 1],
                'Location_Kagera': [0, 0],
                'Location_Katavi': [0, 0],
                'Location_Kigoma': [0, 0],
                'Location_Kilimanjaro': [0, 0],
                'Location_Manyara': [0, 0],
                'Location_Mbeya': [0, 0],
                'Location_Morogoro': [0, 0],
                'Location_Mwanza': [0, 0],
                'Location_Pwani':[0,0],
                'Location_Rukwa':[0.0],
                'Location_Singida':[0.0],
                'Location_Songea':[0.0],
                'Location_Tabora':[0.0],
                'Location_Tanga':[0.0],
                'Location_Zanzibar':[0.0]
               }

        # Convert the dictionary to Pandas dataframe
        data_frame = pd.DataFrame(data)
        df_encoded = pd.get_dummies(data_frame, columns=['Event name', 'Location'])
        df_encoded=df_encoded.astype(np.float32)
        df_encoded
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
    
