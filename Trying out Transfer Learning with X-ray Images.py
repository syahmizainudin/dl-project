<<<<<<< HEAD
"""
Trying out Transfer Learning with X-ray Images
Making a model that can classify x-ray images between normal lungs, one with covid, and one with viral pneumonic infection.

Dataset link: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
"""

# %%
# 1. Import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os, datetime

# %%
# 2. Data loading
DATASET_DIR = 'datasets/Covid19-dataset'
train_path = os.path.join(DATASET_DIR, 'train')
test_path = os.path.join(DATASET_DIR, 'test')

# %%
#3. Data preparation
BATCH_SIZE = 16
IMG_SIZE = (160,160)

# Load the data as tensorflow dataset
train_dataset = keras.utils.image_dataset_from_directory(train_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(test_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# %%
# 4. Display some image from the dataset
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %%
# 5. Convert the train and test dataset into prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# 6. Data augmentation layer
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2)
])

# %%
# 7. Result of multiple data augmentation on an image
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
# 8. Data normalization layer
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# %%
# 9. Start transfer learning
# (A) Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# %%
# (B) Classifier
# Global average pooling layer and a dense layer
global_avg = keras.layers.GlobalAveragePooling2D()
dense = keras.layers.Dense(256, activation='relu')

# Output layer
output_layer = keras.layers.Dense(len(class_names), activation='softmax')

# %%
# 10. Linking the layers to form a pipeline
inputs = keras.layers.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = dense(x)
outputs = output_layer(x)

# Instantiate the full model pipeline
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

# %%
# 11. Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# %%
# 12. Configure callbacks
base_log_path = 'tensorboard_logs/transfer_learning_log'
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(log_dir=log_path)

# %%
# 13. Model training
EPOCHS = 10
history = model.fit(pf_train, validation_data=pf_test, epochs=EPOCHS, callbacks=[tb])

# %%
# 14. Evaluate the model
loss0,acc0 = model.evaluate(pf_test)

print("-"*10, 'Model evaluation after training', "-"*10)
print('Loss =', loss0)
print('Accuracy =', acc0)

# %%
# 15. Model deployment
# Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)

# Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch, y_pred)))

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print('Confusion matrix:\n', confusion_matrix(y_true=label_batch, y_pred=y_pred), '\nClassification report:\n',classification_report(y_true=label_batch, y_pred=y_pred))

# %%
=======
"""
Trying out Transfer Learning with X-ray Images
Making a model that can classify x-ray images between normal lungs, one with covid, and one with viral pneumonic infection.

Dataset link: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
"""

# %%
# 1. Import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os, datetime

# %%
# 2. Data loading
DATASET_DIR = 'datasets/Covid19-dataset'
train_path = os.path.join(DATASET_DIR, 'train')
test_path = os.path.join(DATASET_DIR, 'test')

# %%
#3. Data preparation
BATCH_SIZE = 16
IMG_SIZE = (160,160)

# Load the data as tensorflow dataset
train_dataset = keras.utils.image_dataset_from_directory(train_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(test_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# %%
# 4. Display some image from the dataset
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %%
# 5. Convert the train and test dataset into prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# 6. Data augmentation layer
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2)
])

# %%
# 7. Result of multiple data augmentation on an image
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
# 8. Data normalization layer
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# %%
# 9. Start transfer learning
# (A) Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# %%
# (B) Classifier
# Global average pooling layer and a dense layer
global_avg = keras.layers.GlobalAveragePooling2D()
dense = keras.layers.Dense(256, activation='relu')

# Output layer
output_layer = keras.layers.Dense(len(class_names), activation='softmax')

# %%
# 10. Linking the layers to form a pipeline
inputs = keras.layers.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = dense(x)
outputs = output_layer(x)

# Instantiate the full model pipeline
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

# %%
# 11. Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# %%
# 12. Configure callbacks
base_log_path = 'tensorboard_logs/transfer_learning_log'
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(log_dir=log_path)

# %%
# 13. Model training
EPOCHS = 10
history = model.fit(pf_train, validation_data=pf_test, epochs=EPOCHS, callbacks=[tb])

# %%
# 14. Evaluate the model
loss0,acc0 = model.evaluate(pf_test)

print("-"*10, 'Model evaluation after training', "-"*10)
print('Loss =', loss0)
print('Accuracy =', acc0)

# %%
# 15. Model deployment
# Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)

# Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch, y_pred)))

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print('Confusion matrix:\n', confusion_matrix(y_true=label_batch, y_pred=y_pred), '\nClassification report:\n',classification_report(y_true=label_batch, y_pred=y_pred))

# %%
>>>>>>> eaee0d03039b50e033e0f3400f8133787ff9e1b8
