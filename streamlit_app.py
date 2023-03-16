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

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.beta_columns(13)
with col1:
    text1 = st.text_input("Text 1", value="")
with col2:
    text2 = st.text_input("Text 2", value="")
with col3:
    text3 = st.text_input("Text 3", value="")
with col4:
    text4 = st.text_input("Text 4", value="")
with col5:
    text5 = st.text_input("Text 5", value="")
with col6:
    text6 = st.text_input("Text 6", value="")
with col7:
    text7 = st.text_input("Text 7", value="")
with col8:
    text8 = st.text_input("Text 8", value="")
with col9:
    text9 = st.text_input("Text 9", value="")
with col10:
    text10 = st.text_input("Text 10", value="")
with col11:
    text11 = st.text_input("Text 11", value="")
with col12:
    text12 = st.text_input("Text 12", value="")
with col13:
    text13 = st.text_input("Text 13", value="")

# Check if a file has been uploaded
if csv_file is not None:
    # Load CSV file into a Pandas DataFrame
    
    data_frame= pd.read_csv(csv_file)

    # Show the DataFrame in the app
    st.write("Original DataFrame:")
    st.write(data_frame)
    data_frame=data_frame.drop(index=data_frame.index[-1])
    df_encoded = pd.get_dummies(data_frame, columns=['Event name', 'Location'])
    df_encoded=df_encoded.astype(np.float32)
    df_encoded.plot(kind='scatter',figsize=(15,10),x="Estimated Cost ($)",y='Venue cost')
    plt.grid()
    plt.show()
    
    # Define hyperparameters
    input_dim = 51  # number of input variables
    output_dim = 1  # number of output variables
    hidden_dim = 30  # number of neurons in the hidden layer
    num_epochs = 50  # number of epochs to train the model
    batch_size = 100  # size of batch for each epoch
    learning_rate = 0.01  # learning rate for the optimizer
    
        # Split the data into input (X) and output (y) variables
    X = df_encoded.drop(columns='Estimated Cost ($)')
    y = df_encoded['Estimated Cost ($)']

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    
        # Create the model architecture
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=input_dim, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mape', optimizer=Adam(lr=learning_rate), metrics=['mape'])

    # Train the model on the training data
    history = model.fit(X_train, y_train,epochs=num_epochs,batch_size=batch_size, validation_data=(X_val, y_val))
    #predict values using the trained model
    predictions = model.predict(X_test)
    
    y_test=np.array(y_test)
    plt.plot(predictions)
    plt.plot(y_test)

    # set the axis labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predicted value against real value')
    # display the plot
    plt.show()
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
#     X_test=X_test.head()
    predictions = model.predict(X_test)
#     st.write("Predicted values")
#     st.write(predictions)
#     st.write("Original values")
#     st.write(y_test)
    
    col1, col2 = st.beta_columns(2)
    with col1:
         st.write("Original values")
         st.write(y_test)

    with col2:
        st.write("Predicted values")
        st.write(predictions)
    
