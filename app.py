import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.title("Sales Trend Forecaster (LSTM)")
st.write("This application forecasts future sales trends using an LSTM model.")

@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    return df

df = load_data()
st.subheader("Previous Sales Data")
st.line_chart(df.set_index("Date")["Sales"])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Sales"].values.reshape(-1, 1))

look_back = 7
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back, 0])
    y.append(scaled_data[i + look_back, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))



@st.cache_resource
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    with st.spinner("Training the LSTM model..."):
        model.fit(X_train, y_train, epochs=5, batch_size=20, verbose=1)
    return model


model = train_model(X, y)

if st.button("Predict Future Sales"):
    # Get predictions for the data we have
    predictions_scaled = model.predict(X)
    # Convert predictions back to normal numbers (undo the 0-1 scaling)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Create a plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Actual data (skip the first 7 days used for looking back)
    actual_sales = df["Sales"].values[look_back:]
    
    ax.plot(actual_sales, label="Actual Sales", color='blue')
    ax.plot(predictions, label="AI Predicted Sales", color='red', linestyle='--')
    ax.legend()
    
    st.pyplot(fig)
    st.success("Notice how the red line (AI) learns to follow the blue line (Actual) peaks and drops!")