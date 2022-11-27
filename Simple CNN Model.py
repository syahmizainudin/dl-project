"""
Simple CNN Model

I am creating an image classifiation model for CIFAR10 dataset.
"""

#%%
# 1. Import the required packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, callbacks
import matplotlib.pyplot as plt

#%%
# 2. Data loading
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# %%
# 3. Plotting the data for checking
plt.figure(figsize=(10,10))

for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(y_train[i][0])

plt.show()

# %%
# 4. Normalize pixel values
X_train, X_test = X_train/255.0, X_test/255.0

# %%
# 5. Model creation
img_shape = X_train[0].shape
nClass = len(np.unique(y_train))

# Create the CNN
model = keras.Sequential()

# Input layer
model.add(layers.InputLayer(input_shape=img_shape))

# Feature extractor
model.add(layers.Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D())

# Classifier
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

# Output layer (softmax)
model.add(layers.Dense(nClass, activation='softmax'))

model.summary()

# %%
# 6. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# 7. Create callback function
import os, datetime

file_path = 'tensorboard_logs/cnn_test_logs'
LOG_DIR = os.path.join(file_path, "log_" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=LOG_DIR)
es = callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

# %%
# 8. Perform model training
BATCH_SIZE = 32
EPOCHS = 10

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tb, es])

# %%
# 9. Evaluate the model
score = model.evaluate(X_test, y_test)

# %%
