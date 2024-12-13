import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

# Load model
model = tf.keras.models.load_model('models/model_filtered_6_1.keras')

# try different sizes?
img_height = 128
img_width = 128
batch_size = 16

test_data_dir = pathlib.Path('filtered_data')

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False  
)

class_names = test_ds.class_names
print("Class names:", class_names)

test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

y_pred = model.predict(test_ds)
y_pred_classes = (y_pred > 0.5).astype("int32")  
y_true = np.concatenate([y for _, y in test_ds])


cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix on New Test Data')
plt.show()

accuracy = np.mean(y_true == y_pred_classes)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

plt.figure(figsize=(10, 10))

misclassified_images = []
for image_batch, label_batch in test_ds:
    predictions = model.predict(image_batch)
    predicted_labels = (predictions > 0.5).astype("int32")

    for i in range(len(image_batch)):
        if label_batch[i] != predicted_labels[i]:  
            misclassified_images.append((image_batch[i], label_batch[i], predicted_labels[i]))

 
random.shuffle(misclassified_images)
for j in range(min(9, len(misclassified_images))):
    img, true_label, pred_label = misclassified_images[j]
    plt.subplot(3, 3, j + 1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label[0]]}")
    plt.axis('off')

plt.show()
