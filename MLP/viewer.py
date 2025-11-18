# viewer.py
import numpy
import os

data = numpy.load("dataset.npz")
x, y = data["x"], data["y"]

i = 0
while True:
    os.system("clear")

    for row in x[i].reshape(20, 20):
        print("".join("██" if px == 1 else "  " for px in row))
    print(y[i])

    # input()
    i = (i + 1) % len(x)