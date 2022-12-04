<<<<<<< HEAD
"""
Experimenting with Overfitting Model

1. Applying the below steps to obseve its effects on overfitting model:
    a. Implementing early stopping
    b. Implementing regularization
    c. Implementing dropout
    d. Reducing the numbers of hidden layers and nodes
    e. Using a different type of categorical encoding for the data
"""

#%%
# 1. Import the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# %%
# 2. Data loading
df = pd.read_csv('datasets/germanCredit.csv', delimiter=' ', header=None, names=[
        'ECA', 'M', 'CH', 'P', 'CA', 'SA', 'PEMP', 'IRP', 'PSS', 'DG', 'PRES', 
        'PROP', 'AGE', 'OIP', 'HOU', 'EXCRED', 'JOB', 'LPEOP', 'TEL', 'FORW', 'RISK'
    ])

# %% 
# 3. EDA & Data Cleaning
df.info()

#%%
## 3.1 Check for duplicates
df.duplicated().sum()

# %%
## 3.2 Check for null values
df.isna().sum()

# %%
# 4. Preprocessing
SEED = 12345

## 4.1. Splitting labels and features
# df = pd.concat([df[df['RISK'] == 1][:276], df[df['RISK'] == 2]], axis=0)
features = df.drop('RISK', axis=1)
labels = df['RISK'] - 1

## 4.2 Encode categorical data in the features
from sklearn.preprocessing import LabelEncoder

encode_columns = features.dtypes[features.dtypes == 'object'].index.values.tolist()
label_encoder = LabelEncoder()

# features = pd.get_dummies(features, columns=None, drop_first=True)
features[encode_columns] = features[encode_columns].apply(label_encoder.fit_transform)

# %%
## 4.3 Perform train test split for train and test data
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, 
                                                    test_size=0.3, random_state=SEED)

# %%
## 4.4 Create normalization object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
## 4.5 Doing data normalization
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# 5. Create a model
nIn = X_train.shape[1]
nClass = len(np.unique(labels))
l1 = keras.regularizers.L1()
l2 = keras.regularizers.L2()

model = keras.Sequential()
model.add(keras.layers.InputLayer(nIn,))
# model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
# model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=l2))
model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
model.add(keras.layers.Dense(nClass, activation='softmax'))

model.summary()

# %%
# 6. Compile the model
from tensorflow_addons.metrics import RSquare
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# 7. Create the callback function
import os, datetime

# Early stopping callback function
es = keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

# TensorBoard callback function
LOG_DIR = 'tensorboard_logs/exercise_4_logs'
LOG_DIR = os.path.join(LOG_DIR, "log_" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 
                        + ' - ES, L2, LAYERS, NODES, LABEL, DROPOUT_1_03')
tb = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# %%
# 8. Model training
BATCH_SIZE = 64
EPOCHS = 100
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, callbacks=[tb, es])

# %%
# 9. Model evaluation
evaluation = model.evaluate(X_test, y_test)

# %%
# 10. Visualizing model training process
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_no = history.epoch

# Plotting epoch againts loss
plt.plot(epochs_no, training_loss, label='Training Loss')
plt.plot(epochs_no, val_loss, label='Validation Loss')
plt.legend()
plt.show()

#Plotting epoch againts accuracy
plt.plot(epochs_no, training_acc, label='Training Accuracy')
plt.plot(epochs_no, val_acc, label='Validation Accuracy')
plt.legend()
plt.show()

# %% 
# 11. Making prediction with the model
from sklearn.metrics import confusion_matrix, classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_pred, y_test), '\n'
        , classification_report(y_pred, y_test))
print(labels.value_counts())
# %%
=======
"""
Experimenting with Overfitting Model

1. Applying the below steps to obseve its effects on overfitting model:
    a. Implementing early stopping
    b. Implementing regularization
    c. Implementing dropout
    d. Reducing the numbers of hidden layers and nodes
    e. Using a different type of categorical encoding for the data
"""

#%%
# 1. Import the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# %%
# 2. Data loading
df = pd.read_csv('datasets/germanCredit.csv', delimiter=' ', header=None, names=[
        'ECA', 'M', 'CH', 'P', 'CA', 'SA', 'PEMP', 'IRP', 'PSS', 'DG', 'PRES', 
        'PROP', 'AGE', 'OIP', 'HOU', 'EXCRED', 'JOB', 'LPEOP', 'TEL', 'FORW', 'RISK'
    ])

# %% 
# 3. EDA & Data Cleaning
df.info()

#%%
## 3.1 Check for duplicates
df.duplicated().sum()

# %%
## 3.2 Check for null values
df.isna().sum()

# %%
# 4. Preprocessing
SEED = 12345

## 4.1. Splitting labels and features
# df = pd.concat([df[df['RISK'] == 1][:276], df[df['RISK'] == 2]], axis=0)
features = df.drop('RISK', axis=1)
labels = df['RISK'] - 1

## 4.2 Encode categorical data in the features
from sklearn.preprocessing import LabelEncoder

encode_columns = features.dtypes[features.dtypes == 'object'].index.values.tolist()
label_encoder = LabelEncoder()

# features = pd.get_dummies(features, columns=None, drop_first=True)
features[encode_columns] = features[encode_columns].apply(label_encoder.fit_transform)

# %%
## 4.3 Perform train test split for train and test data
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, 
                                                    test_size=0.3, random_state=SEED)

# %%
## 4.4 Create normalization object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
## 4.5 Doing data normalization
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# 5. Create a model
nIn = X_train.shape[1]
nClass = len(np.unique(labels))
l1 = keras.regularizers.L1()
l2 = keras.regularizers.L2()

model = keras.Sequential()
model.add(keras.layers.InputLayer(nIn,))
# model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
# model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=l2))
# model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=l2))
model.add(keras.layers.Dropout(rate=0.3, seed=SEED))
model.add(keras.layers.Dense(nClass, activation='softmax'))

model.summary()

# %%
# 6. Compile the model
from tensorflow_addons.metrics import RSquare
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# 7. Create the callback function
import os, datetime

# Early stopping callback function
es = keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

# TensorBoard callback function
LOG_DIR = 'tensorboard_logs/overfit_test_logs'
LOG_DIR = os.path.join(LOG_DIR, "log_" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 
                        + ' - ES, L2, LAYERS, NODES, LABEL, DROPOUT_1_03')
tb = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# %%
# 8. Model training
BATCH_SIZE = 64
EPOCHS = 100
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, callbacks=[tb, es])

# %%
# 9. Model evaluation
evaluation = model.evaluate(X_test, y_test)

# %%
# 10. Visualizing model training process
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_no = history.epoch

# Plotting epoch againts loss
plt.plot(epochs_no, training_loss, label='Training Loss')
plt.plot(epochs_no, val_loss, label='Validation Loss')
plt.legend()
plt.show()

#Plotting epoch againts accuracy
plt.plot(epochs_no, training_acc, label='Training Accuracy')
plt.plot(epochs_no, val_acc, label='Validation Accuracy')
plt.legend()
plt.show()

# %% 
# 11. Making prediction with the model
from sklearn.metrics import confusion_matrix, classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_pred, y_test), '\n'
        , classification_report(y_pred, y_test))
print(labels.value_counts())
# %%
>>>>>>> eaee0d03039b50e033e0f3400f8133787ff9e1b8
