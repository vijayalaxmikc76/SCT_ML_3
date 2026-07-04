Task 3: Support Vector Machine (SVM) - Cats vs Dogs Image Classification

Objective

The objective of this project is to build a Support Vector Machine (SVM) model that classifies images of cats and dogs. The model is trained using image preprocessing techniques and evaluated using standard classification metrics.


Dataset

Dataset Used:
Microsoft Cats vs Dogs (PetImages Dataset)

The dataset contains two classes:
- Cat
- Dog

Corrupted images were automatically skipped during preprocessing.

Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Steps Performed

1. Loaded images from the PetImages dataset.
2. Removed corrupted images.
3. Resized images to 64×64 pixels.
4. Converted images to grayscale.
5. Flattened image pixels into feature vectors.
6. Split the dataset into training and testing sets.
7. Trained a Support Vector Machine (SVM) classifier.
8. Evaluated the model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
9. Displayed sample predictions.

Results

- Successfully classified cat and dog images using SVM.
- Evaluated model performance using multiple evaluation metrics.
- Visualized prediction results.

Learning Outcomes

- Image preprocessing using OpenCV.
- Feature extraction from images.
- Binary image classification using SVM.
- Model evaluation using confusion matrix and classification report.
- Working with real-world datasets containing corrupted images.
