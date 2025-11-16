# dataset.py
from PIL import Image, ImageDraw, ImageFont
import numpy
import os

SYMBOLS = [i for i in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789.,;:!?()-«»"]
FONTS = os.listdir("fonts")

if __name__ == "__main__":
    x, y = [], []
    done = 1

    for symbol in SYMBOLS:
        for font in FONTS:
            font = ImageFont.truetype(os.path.join("fonts", font), 25)
            img = Image.new("L", (40, 40), color=0)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), symbol, font=font, fill=255)
            img = img.crop(img.getbbox())
            img = img.resize((20, 20))

            arr = numpy.array(img)
            arr = (arr > 100).astype(numpy.uint8)
            x.append(arr.flatten())
            y.append(symbol)

            print(f"\r{done}/{len(SYMBOLS) * len(FONTS)}", end="")
            done += 1

    x = numpy.array(x, dtype=numpy.uint8)
    y = numpy.array(y)

    numpy.savez("dataset.npz", x=x, y=y)

    print()
