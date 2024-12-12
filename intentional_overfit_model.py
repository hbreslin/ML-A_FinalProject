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
data_dir = pathlib.Path('filtered_data')
image_count = len(list(data_dir.glob('*/*.jpg')))

discnt = len(list(data_dir.glob('disliked/*.jpg')))
likecnt = len(list(data_dir.glob('liked/*.jpg')))

print(discnt)
print(likecnt)

# Batch size and image size
batch_size = 16
img_height = 244
img_width = 244

# Load dataset
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  # Ensure shuffling is applied
)

# Get class names before applying prefetch
class_names = train_ds.class_names
print("Class names:", class_names)

# Prefetch data to optimize performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Compute class weights
print('Going into computing weights')
y_train = []
for image_batch, label_batch in train_ds:
    y_train.extend(label_batch.numpy())

print("Label distribution:", Counter(y_train))  # Check the class distribution

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Data augmentation layer (minimized for overfitting)
# print('Going into making aug layer')
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),  # Reduced the rotation range
#     layers.RandomZoom(0.1),  # Reduced zoom range
#     layers.RandomContrast(0.1),  # Reduced contrast change
#     layers.RandomBrightness(0.1)  # Reduced brightness change
# ])

# Define the model with increased complexity (more filters)
print('Going into model definition')

def make_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)  # Increased number of filters
    x = layers.MaxPooling2D()(x)

    # Additional convolutional layers with more filters
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)  # Increased filters
    x = layers.MaxPooling2D()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Optional: remove dropout or reduce its rate
    x = layers.Dropout(0.2)(x)  # Reduced dropout rate

    # Output layer with sigmoid activation for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)


# v2
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def make_model(input_shape):
#     inputs = tf.keras.Input(shape=input_shape)

#     # Entry block
#     x = layers.Rescaling(1.0 / 255)(inputs)  # Normalize pixel values to [0,1]
    
#     # Convolutional layers with increased filters
#     x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)  # Increased filters from 64 to 128
#     x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)  # Added another conv layer (deeper)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Slightly reduce dimensions

#     x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)  # Increased filters from 128 to 256
#     x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)  # Added another conv layer (deeper)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)  # Even more filters (512)
#     x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)  # Another convolutional layer
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Fully connected layers (Global pooling removed to allow for memorization)
#     x = layers.Flatten()(x)  # Flatten instead of GlobalAveragePooling to retain all parameters
#     x = layers.Dense(1024, activation="relu")(x)  # Fully connected layer with lots of neurons
#     x = layers.Dense(512, activation="relu")(x)  # Additional dense layer
#     # **REMOVED DROPOUT** for intentional overfitting

#     # Output layer with sigmoid activation for binary classification
#     outputs = layers.Dense(1, activation="sigmoid")(x)

#     return models.Model(inputs, outputs)


# Define input shape and model
input_shape = (244, 244, 3)  # Example image size (244x244 RGB images)
model = make_model(input_shape)

# Compile the model with a lower learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,  # Higher learning rate for faster learning
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

print('Going into compiling')
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

# Early stopping callback
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=3, restore_best_weights=True
# )

# Train the model with class weights
epochs = 15  # Training for more epochs to allow overfitting
print('Going into fitting')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight_dict,
    # callbacks=[early_stopping],  # Early stopping to avoid excessive overfitting
    verbose=1 
)

# Save the model
model.save('models/intentional_overfit_2_1.keras')

# Plot training history
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

# Predict on validation dataset
y_pred = model.predict(val_ds)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary labels (0 or 1)

# Get true labels
y_true = np.concatenate([y for _, y in val_ds])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
