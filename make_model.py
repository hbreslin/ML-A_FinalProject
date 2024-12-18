# sources: https://keras.io/examples/vision/image_classification_from_scratch/

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pathlib
import os
import keras
from tensorflow import data as tf_data

# Path to dataset
data_dir = pathlib.Path('sorted_data')
image_count = len(list(data_dir.glob('*/*.jpg')))

discnt = len(list(data_dir.glob('disliked/*.jpg')))
likecnt = len(list(data_dir.glob('liked/*.jpg')))

print(discnt)
print(likecnt)

# Batch size and image size
batch_size = 32
img_height = 128
img_width = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  # make sure shuffled
)

class_names = train_ds.class_names
print("Class names:", class_names)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print('Going into computing weights')
y_train = []
for image_batch, label_batch in train_ds:
    y_train.extend(label_batch.numpy())

print("Label distribution:", Counter(y_train))  # Check the class distribution?

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# print('Going into making aug layer')
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(0.2),
#     layers.RandomContrast(0.2),
#     layers.RandomBrightness(0.2)
# ])

num_classes = len(class_names)
print('Going into model definition')

def make_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x) #added
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)

input_shape = (img_width, img_height, 3) 
model = make_model(input_shape)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,  
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

print('Going into compiling')
model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

epochs = 30
print('Going into fitting')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1 
)

# Save the model
model.save('models/model_filtered_6_1.keras')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(val_ds)
y_pred_classes = (y_pred > 0.5).astype("int32")  

y_true = np.concatenate([y for _, y in val_ds])

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
