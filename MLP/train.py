# train.py
import tensorflow
import numpy

data = numpy.load("dataset.npz")
x, y = data["x"], data["y"]

symbols = numpy.unique(y)
symbol_to_idx = {s: i for i, s in enumerate(symbols)}

y = numpy.array([symbol_to_idx[s] for s in y])
num_classes = len(symbols)
y = tensorflow.keras.utils.to_categorical(y, num_classes)

indices = numpy.arange(len(x))
numpy.random.shuffle(indices)
x = x[indices]
y = y[indices]

split = int(0.8 * len(x))
X_train, X_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Input(shape=(400,)),
    tensorflow.keras.layers.Dense(512, activation="relu"),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dropout(0.5),
    tensorflow.keras.layers.Dense(256, activation="relu"),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dropout(0.5),
    tensorflow.keras.layers.Dense(128, activation="relu"),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dropout(0.4),
    tensorflow.keras.layers.Dense(64, activation="relu"),
    tensorflow.keras.layers.Dropout(0.3),
    tensorflow.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1, 
    patience=3, 
    min_lr=0.000001
)

early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=20, 
    restore_best_weights=True
)

model.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=64,
    shuffle=True,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
    )

model.save("model.keras")
numpy.savez("labels.npz", symbols=symbols)
