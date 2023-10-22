#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,
cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt
iris=datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
 columns = iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']
model = LinearDiscriminantAnalysis()
model.fit(X, y)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))
X = iris.data
y = iris.target
model = LinearDiscriminantAnalysis()
data_plot = model.fit(X, y).transform(X)
target_names = iris.target_names
plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
 plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
 label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()
# define new observation
new = [5, 2, 1, .4]
# predict which class the new observation belongs to
model.predict([new])

