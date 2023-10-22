#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns


# In[2]:


df = pd.read_csv('uber.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.dropna(subset=['dropoff_longitude','dropoff_latitude'], inplace=True)


# In[5]:


df.isnull().sum()


# In[25]:


sns.boxplot(x=df['fare_amount'])


# In[7]:


correlation = df.corr()
sns.heatmap(correlation
            


# In[10]:


x = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = df['fare_amount']


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[12]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[13]:


lr_model = LinearRegression()
lr_model.fit(x_train, y_train)


# In[15]:


ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)


# In[16]:


lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)


# In[17]:


lr_pred = lr_model.predict(x_test)
ridge_pred = ridge_model.predict(x_test)
lasso_pred = lasso_model.predict(x_test)


# In[20]:


lr_r2 = r2_score(y_test, lr_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)


# In[21]:


lr_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))


# In[22]:


print("Linear Regression R-squared:", lr_r2)
print("Ridge Regression R-squared:", ridge_r2)
print("Lasso Regression R-squared:", lasso_r2)


# In[23]:


print("Linear Regression RMSE:", lr_rmse)
print("Ridge Regression RMSE:", ridge_rmse)
print("Lasso Regression RMSE:", lasso_rmse)


# In[ ]:




