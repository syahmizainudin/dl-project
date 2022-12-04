"""
A Simple Model using Functional API

1. I am making an MLP model using Functional API with 4 hidden layers.
"""
#%%
# 1. Import required packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow import keras
import tensorflow_addons as tfa

#%%
# 2. Load the data
diabetes = load_diabetes()
features = diabetes.data
labels = diabetes.target

# %%
# 3. EDA & Data cleaning
## I then explore the data and do some data cleaning
df = pd.DataFrame(data=features, columns=diabetes.feature_names)

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
## 4.2 Normalization --> The data had already been normalized

# %%
# 5. Model creation with functional API
nIn = X_train.shape[1]

## 5.1 Input layer
inputs = keras.Input(shape=(nIn,))

## 5.2 Hidden layer
n_nodes = 512
hidden_layers = []

for i in range(0, 4):
    hidden_layer = keras.layers.Dense(n_nodes/2**i, activation='relu')
    hidden_layers.append(hidden_layer)

## 5.3 Output layer --> Output layer is set to 1 node as the model is a Regression model
out_layer = keras.layers.Dense(1)

# 5.4 Use the functional API
x = hidden_layers[0](inputs)

for layer in hidden_layers[1:]:
    x = layer(x)

outputs = out_layer(x)

# 5.5 Instantiate the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
# %%
# 6. Model compilation
rmse = keras.metrics.RootMeanSquaredError()
r_squared = tfa.metrics.RSquare()

model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse, r_squared])

# %%
# 7. Model training
BATCH_SIZE = 32
EPOCHS = 30

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# %%
# 8. Model evaluation
evaluation = model.evaluate(X_test, y_test)

# %%
# 9. Visualize training result
import matplotlib.pyplot as plt

train_loss, train_mae, train_rmse, train_r2, val_loss, val_mae, val_rmse, val_r2 = history.history.values()
epochs_no = history.epoch

## 9.1 Plot epoch againts loss (MSE)
plt.subplot(2,2,1)
plt.plot(epochs_no, train_loss, label='Training Loss')
plt.plot(epochs_no, val_loss, label='Validation Loss')
plt.legend()

## 9.1 Plot epoch againts MAE
plt.subplot(2,2,2)
plt.plot(epochs_no, train_mae, label='Training MAE')
plt.plot(epochs_no, val_mae, label='Validation MAE')
plt.legend()

## 9.2 Plot epoch againts RMSE
plt.subplot(2,2,3)
plt.plot(epochs_no, train_rmse, label='Training RMSE')
plt.plot(epochs_no, val_rmse, label='Validation RMSE')
plt.legend()

## 9.2 Plot epoch againts R2
plt.subplot(2,2,4)
plt.plot(epochs_no, train_r2, label='Training R2')
plt.plot(epochs_no, val_r2, label='Validation R2')
plt.legend()

plt.show()

# %%
# 10. Making prediction with the model
y_pred = model.predict(X_test)
# %%
## 10.1 Scatter plot of the prediction againts its label
plt.scatter(y_pred, y_test)
plt.show()
# %%
