import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Dataset Path
# ==========================

dataset_path = r"C:\Users\VIAJAYLAXMI\OneDrive\Desktop\python\PetImages"

categories = ["Cat", "Dog"]

IMG_SIZE = 64
MAX_IMAGES = 25000      # 1000 Cats + 1000 Dogs

data = []
labels = []

# ==========================
# Load Images
# ==========================

for category in categories:

    path = os.path.join(dataset_path, category)

    count = 0

    print(f"\nReading {category} images...")

    for image_name in os.listdir(path):

        if count >= MAX_IMAGES:
            break

        try:

            img_path = os.path.join(path, image_name)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # HOG Feature Extraction
            features = hog(
                image,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm='L2-Hys'
            )

            data.append(features)

            if category == "Cat":
                labels.append(0)
            else:
                labels.append(1)

            count += 1

        except:
            continue

    print(f"{category}: {count} images loaded")

# ==========================
# Convert to NumPy
# ==========================

X = np.array(data)
y = np.array(labels)

print("\nTotal Images:", len(X))

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Images:", len(X_train))
print("Testing Images :", len(X_test))

# ==========================
# Train SVM
# ==========================

print("\nTraining SVM...")

model = LinearSVC(max_iter=10000)

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================
# Prediction
# ==========================

predictions = model.predict(X_test)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy*100:.2f}%")

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix")

print(cm)

# ==========================
# Classification Report
# ==========================

print("\nClassification Report")

print(classification_report(y_test, predictions))

# ==========================
# Display Predictions
# ==========================

plt.figure(figsize=(12,8))

for i in range(6):

    index = np.random.randint(0, len(X_test))

    img = cv2.imread(os.path.join(
        dataset_path,
        categories[y_test[index]],
        os.listdir(os.path.join(dataset_path, categories[y_test[index]]))[0]
    ), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (64,64))

    plt.subplot(2,3,i+1)

    plt.imshow(img, cmap='gray')

    predicted = "Cat" if predictions[index]==0 else "Dog"
    actual = "Cat" if y_test[index]==0 else "Dog"

    plt.title(f"Pred: {predicted}\nActual: {actual}")

    plt.axis("off")

plt.tight_layout()

plt.show()