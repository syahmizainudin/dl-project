"""
Predicting Boston House Price Using K-Fold Cross Validation with Neural Network 
"""
#%%
# 1. Import the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras

# %%
# 2. Data loading
df = pd.read_csv('datasets/boston.csv')

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

#%%
## 4.1 Splitting labels and features
features = df.drop('MEDV', axis=1).values
labels = df['MEDV'].values

#%%
## 4.2 Create KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# %%
## 4.3 Create normalization object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
# 5. Model creation
nIn = features.shape[1]

## 5.1 Creating a function for model creation
def create_model():
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(nIn,)),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1),
        ]
    )
    return model

# %%
# 6. Run the KFold
import tensorflow_addons as tfa

fold_no = 1
loss_list = []
r_square_list = []
BATCH_SIZE = 128
EPOCHS = 50

for train_idx, test_idx in kf.split(features, labels):
    ## 6.1 Splitting the data into train and test data
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    ## 6.2 Doing data normalization
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ## 6.3 Create the model
    model = create_model()

    ## 6.4 Compile the model
    r_square = tfa.metrics.RSquare()
    model.compile(optimizer='adam', loss='mse', metrics=[r_square])

    ## 6.5 Train the model
    print("#"*50)
    print(f'Training KFold: {fold_no}')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)
    print("#"*50)


    ## 6.6 Model evaluation
    scores = model.evaluate(X_test, y_test)

    ## 6.7 Append scores to list
    for metric_name, score in zip(model.metrics_names, scores):
        print(f'{metric_name}: {score}')

    loss_list.append(scores[0])
    r_square_list.append(scores[1])

    ## 6.8 Initiating next fold and clearing the memory for next model
    fold_no+=1
    keras.backend.clear_session()
# %%
# 7. Getting the average loss and r_square from the cross valiadtion results
mean_loss = np.mean(np.array(loss_list))
mean_r_square = np.mean(np.array(r_square_list))

print(f'Mean Loss: {mean_loss}\nMean R-squared: {mean_r_square}')

# %%
# 8. Visualize the score of each fold
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(range(1, len(loss_list)+1), loss_list)
plt.title('Loss for Each Fold')

plt.subplot(1,2,2)
plt.plot(range(1, len(r_square_list)+1), r_square_list)
plt.title('R-Square for Each Fold')

plt.show()

# %%
# 9. Making prediction with the model and plotting prediction againts the target
pred = model.predict(X_test)
plt.scatter(pred, y_test)
plt.title('Prediction Againts Target')
plt.show()
# %%
