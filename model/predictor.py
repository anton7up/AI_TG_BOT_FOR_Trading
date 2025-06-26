import numpy as np
from tensorflow.keras.models import load_model
from data.fetch_data import fetch_data
from config import TICKER

def predict_next_day():
    model = load_model('models/lstm_AAPL_model.h5')
    df = fetch_data(TICKER)
    series = df['Close'].values.reshape(-1, 1)
    last_sequence = series[-10:].reshape((1, 10, 1))  # последняя 10-дневная последовательность
    prediction = model.predict(last_sequence)
    return float(prediction[0][0])
