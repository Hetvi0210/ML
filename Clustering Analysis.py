#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install threadpoolctl')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[3]:


# Load the Iris dataset from a CSV file
data = pd.read_csv(r'C:\Users\hp\Desktop\Iris.csv')

# Select the features (attributes) for clustering
X = data.iloc[:, [0, 1, 2, 3]].values

# Standardize the data (mean=0, std=1) to ensure equal weight for all features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[6]:


import os
os.environ['OMP_NUM_THREADS'] = '1'


# In[7]:


wcss = []  # List to store the sum of squared distances for different values of k

# Use a loop to fit K-Means for a range of k values
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid()
plt.show()


# In[8]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)

