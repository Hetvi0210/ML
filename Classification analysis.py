#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
# Load the MNIST dataset
digits = datasets.load_digits()
# Split the dataset into features (X) and labels (Y)
X = digits.data
Y = digits.target
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=42)
# Create an SVM classifier
clf = svm.SVC(kernel='linear')
# Train the classifier on the training data
clf.fit(X_train, Y_train)
# Make predictions on the test data
Y_pred = clf.predict(X_test)
# Evaluate the classifier's performance
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)
# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)
# Visualize some of the misclassified digits
misclassified = np.where(Y_test != Y_pred)[0]
plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified[:10]):
 plt.subplot(2, 5, i + 1)
 plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
 plt.title(f"True: {Y_test[idx]}\nPredicted: {Y_pred[idx]}")
 plt.axis('off')
plt.tight_layout()
plt.show()

