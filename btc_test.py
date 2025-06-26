import os
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def main():
    print(f"Текущая рабочая директория: {os.getcwd()}")

    # Загружаем модель
    model_path = 'model/models/lstm_AAPL_model.h5'
    model = load_model(model_path, compile=False)
    print("Модель загружена.")

    # Загружаем данные BTC за последний месяц (пример)
    btc_data = yf.download('BTC-USD', period='30d', interval='1d')
    print(f"Загружено {len(btc_data)} строк данных.")

    # Берем только колонку Close (как в обучении)
    prices = btc_data['Close'].values.reshape(-1, 1)

    # Нормализуем данные как при обучении
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Создаем окно для последнего дня
    window_size = 30  # должно совпадать с размером окна при обучении
    x_test = []
    x_test.append(prices_scaled[-window_size:])
    x_test = np.array(x_test)

    # Предсказываем
    predicted_scaled = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted_scaled)

    predicted_price = predicted[0, 0]
    real_price = float(prices[-1])

    print(f"Предсказанная цена BTC за последний день: {predicted_price:.2f}")
    print(f"Реальная цена BTC за последний день: {real_price:.2f}")

if __name__ == "__main__":
    main()
