
# ADVANCED IMAGE CLASSIFICATION (MNIST DATASET)
# Using Convolutional Neural Networks (CNN)

# Step 1️⃣: Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Step 2️: Load the MNIST Dataset
print("\n Loading MNIST dataset...")
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")

# Step 3️: Data Preprocessing
# Normalize pixel values (0–1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to fit CNN input: (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to categorical (one-hot encoding)
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Step 4️: Visualize Sample Images
plt.figure(figsize=(8, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.show()

# Step 5️: Build an Advanced CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(num_classes, activation='softmax')
])

# Step 6️: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 7️: Model Summary
print("\n Model Summary:")
model.summary()

# Step 8️: Train the Model
print("\n Training the CNN Model...\n")
history = model.fit(
    x_train, y_train_cat,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Step 9️: Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%")

# Step 10: Plot Training and Validation Curves
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Step 11️: Generate Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Step 12️: Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 13️: Classification Report
print("\n Classification Report:\n")
print(classification_report(y_true, y_pred_classes))

# Step 14️: Visualize Some Predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[i]}, True: {y_true[i]}")
    plt.axis('off')
plt.suptitle("Sample Predictions (First 10 Images)")
plt.show()

# Step 15️: Save the Model
model.save("advanced_mnist_cnn_model.h5")
print("\n Model saved as 'advanced_mnist_cnn_model.h5'")

