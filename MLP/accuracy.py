# accuracy.py
import tensorflow
import numpy

data = numpy.load("dataset.npz")
x, y = data["x"], data["y"]

model = tensorflow.keras.models.load_model("model.keras")
labels = numpy.load("labels.npz")["symbols"]

predict = model.predict(x, batch_size=64, verbose=0)
predict = labels[numpy.argmax(predict, axis=1)]

for label in labels:
    mask = y == label
    total = mask.sum()
        
    correct = (predict[mask] == y[mask]).sum()
    acc = correct / total * 100
    
    predictions = predict[mask & (predict != y)]
    info, counts = numpy.unique(predictions, return_counts=True)
    
    info_list = []
    for info_label, count in zip(info, counts):
        info_list.append(f"{info_label}: {count / total * 100:.1f}%")
    
    info_list.sort(key=lambda x: float(x.split(": ")[1][:-1]), reverse=True)

    print(f"{label}: {correct}/{total}  -  {acc:.2f}%  ({', '.join(info_list)})")