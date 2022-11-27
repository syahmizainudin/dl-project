"""
A simple MLP for Breast Cancer Prediction

1. I am performing machine learning on breast cancer dataset.
2. I am constructing an MLP neural network to solve this problem.
3. I am using Sequential API to create the model with 3 hidden layers.
"""
#%%
# 1. Import required packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from tensorflow import keras

#%%
# 2. Load the data
cancer = load_breast_cancer()
features = cancer.data
labels = cancer.target

# %%
## As the dataset is a classification problem, I first find the number of classes in the labels
## As the number of class is 2 with a value of 0 and 1, Binary Classification model can be consider
nClass = len(np.unique(labels))

# %%
# 3. EDA & Data cleaning
## I then explore the data and do some data cleaning
df = pd.DataFrame(data=features, columns=cancer.feature_names)

#%%
## 3.1 Check for duplicates
df.duplicated().sum()

# %%
## 3.2 Check for null values
df.isna().sum()

# %%
# 4. Preprocessing
## 4.1 Splitting the data into training data and validation data
from sklearn.model_selection import train_test_split

SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED)

# %%
## 4.2 Normalization --> I am doing data normalization using a normalize layer in ANN
normalize = keras.layers.Normalization()
normalize.adapt(X_train)

# %%
# 5. Model creation with Sequential API
## 5.1 Creating Sequential API
model = keras.Sequential()

## 5.2 Input layer
model.add(normalize)

## 5.3 Hidden layer
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))

## 5.4 Output layer --> Output layer is set to 1 node as the model is a Binary Classification model with 2 classes
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# %%
# 6. Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# %%
# 7. Model training
BATCH_SIZE = 64
EPOCHS = 10

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# %%
# 8. Model evaluation
evaluation = model.evaluate(X_test, y_test)

# %%
# 9. Visualize training result
import matplotlib.pyplot as plt

train_loss, train_acc, val_loss, val_acc = history.history.values()
epochs_no = history.epoch

## 9.1 Plot epoch againts loss
plt.plot(epochs_no, train_loss, label='Training Loss')
plt.plot(epochs_no, val_loss, label='Valuation Loss')
plt.legend()
plt.show()

## 9.2 Plot epoch againts accuracy
plt.plot(epochs_no, train_acc, label='Training Accuracy')
plt.plot(epochs_no, val_acc, label='Valuation Accuracy')
plt.legend()
plt.show()

# %%
# 10. Making prediction with the model
y_pred = model.predict(X_test)

# %%
