import easyocr
import os
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

reader = easyocr.Reader(['ru'])

def group(results):
    if not results:
        return []

    texts, x, y, w, h = [], [], [], [], []

    for coordinate, text, _ in results:
        x_arr = [px[0] for px in coordinate]
        y_arr = [px[1] for px in coordinate]
        texts.append(text.strip())
        x.append(sum(x_arr) / 4.0)
        y.append(sum(y_arr) / 4.0)
        w.append(max(x_arr) - min(x_arr))
        h.append(max(y_arr) - min(y_arr))

    def sort_key_index(i):
        return (y[i], x[i])

    def sort_key_x(j):
        return x[j]

    sort_indexes = sorted(range(len(texts)), key=sort_key_index)

    lines, current_line = [], []

    for i in sort_indexes:
        if not current_line:
            current_line = [i]
            continue
        if abs(y[i] - y[current_line[-1]]) <= np.mean(h) * 0.6:
            current_line.append(i)
        else:
            current_line.sort(key=sort_key_x)
            lines.append(current_line)
            current_line = [i]

    if current_line:
        current_line.sort(key=sort_key_x)
        lines.append(current_line)

    return "\n".join([" ".join(texts[j] for j in line) for line in lines])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь мне изображение, и я распознаю текст на нём.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"dataset/temp_{len(os.listdir('dataset'))}.jpg"
    await file.download_to_drive(file_path)

    results = reader.readtext(file_path)
    text = group(results)

    if text == []:
        await update.message.reply_text("Не удалось распознать текст 😔")
    else:
        await update.message.reply_text(text)

def main():
    telegram = Application.builder().token("TOKEN").build()

    telegram.add_handler(CommandHandler("start", start))
    telegram.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    telegram.run_polling()

if __name__ == "__main__":
    os.system("mkdir dataset")
    main()
