import telebot
from telebot import types
from config import TELEGRAM_TOKEN
from btc_test_2 import predict_next_day_price

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Команда /start — показывает клавиатуру
@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_btc = types.KeyboardButton("📈 BTC")
    markup.add(btn_btc)
    bot.send_message(message.chat.id, "Привет! Выбери криптовалюту для прогноза:", reply_markup=markup)

# Обработка кнопки "📈 BTC"
@bot.message_handler(func=lambda message: message.text == "📈 BTC")
def handle_btc(message):
    bot.send_message(message.chat.id, "🔄 Получаю прогноз для BTC, подожди пару секунд...")
    result = predict_next_day_price()
    bot.send_message(message.chat.id, result)

# Обработка любого другого текста
@bot.message_handler(func=lambda message: True)
def handle_unknown(message):
    bot.send_message(message.chat.id, "Пожалуйста, выбери криптовалюту с клавиатуры. Сейчас доступна только 📈 BTC.")

# Запуск бота
if __name__ == '__main__':
    print("✅ Бот запущен.")
    bot.polling(none_stop=True)
