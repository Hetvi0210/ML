#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[3]:


pip install numpy pandas scikit-learn xgboost


# In[4]:


# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target


# In[5]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Initialize the classifiers
ada_clf = AdaBoostClassifier()
gbm_clf = GradientBoostingClassifier()
xgb_clf = XGBClassifier()


# In[7]:


classifiers = [ada_clf, gbm_clf, xgb_clf]
clf_names = ["AdaBoost", "Gradient Boosting", "XGBoost"]


# In[8]:


# Train and evaluate each classifier
for i, clf in enumerate(classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluation measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # For binary classification, you can calculate ROC-AUC score.
    # However, the Iris dataset has 3 classes, so ROC-AUC may not be applicable here.
    # You can use multi-class metrics like accuracy, precision, recall, and F1-score.

    print(f"Classifier: {clf_names[i]}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print()

