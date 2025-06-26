import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from data.fetch_data import fetch_data
from config import TICKER
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib


def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_model():
    df = fetch_data(TICKER)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("Колонки в df:", df.columns.tolist())

    if df is None or df.empty:
        print("❌ Ошибка: данные не загружены. Проверь интернет и тикер:", TICKER)
        return

    if 'Close' not in df.columns:
        print("❌ В данных отсутствует колонка 'Close'")
        return

    series = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    # Сохраняем scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.save')

    window_size = 30
    X, y = create_sequences(series_scaled, window_size=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'📉 RMSE на тесте: {rmse:.4f}')

    model.save('models/lstm_btc_model.h5')
    print("✅ Модель обучена и сохранена в models/lstm_btc_model.h5")

    # График обучения
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Обучение модели')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train_model()
