import numpy as np
import os
import yfinance as yf
from tensorflow.keras.models import load_model
import requests
import joblib


def get_real_time_btc_price():
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data['bitcoin']['usd']
    except Exception as e:
        return None


def predict_next_day_price() -> str:
    model_path = os.path.join('models', 'lstm_btc_model_v2.h5')
    scaler_path = os.path.join('models', 'scaler.save')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "❌ Не найдена модель или scaler. Сначала обучите модель."

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        return f"❌ Ошибка при загрузке модели или scaler: {e}"

    try:
        btc_data = yf.download('BTC-USD', period='999d', interval='1d')
    except Exception as e:
        return f"❌ Ошибка при загрузке данных с Yahoo Finance: {e}"

    if btc_data.empty:
        return "❌ Не удалось загрузить данные BTC."

    close_prices = btc_data['Close'].values.reshape(-1, 1)
    window_size = 30

    if len(close_prices) < window_size:
        return f"❌ Недостаточно данных для окна размером {window_size}"

    try:
        last_window = close_prices[-window_size:]
        scaled_window = scaler.transform(last_window)
        input_window = scaled_window.reshape((1, window_size, 1))

        predicted_scaled_price = model.predict(input_window)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0][0]
    except Exception as e:
        return f"❌ Ошибка при прогнозировании: {e}"

    real_time_price = get_real_time_btc_price()
    if real_time_price is None:
        return "❌ Не удалось получить текущую цену BTC с CoinGecko."

    percent_change = ((predicted_price - real_time_price) / real_time_price) * 100

    result = (
        f"📈 Предсказанная цена BTC на следующий день: {predicted_price:.2f} USD\n"
        f"💰 Текущая цена BTC (CoinGecko): {real_time_price:.2f} USD\n"
        f"📊 Ожидаемое изменение: {percent_change:.2f}%"
    )
    return result


# Для отладки вручную
if __name__ == '__main__':
    print(predict_next_day_price())
