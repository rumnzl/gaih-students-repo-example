#!/usr/bin/env python
# coding: utf-8

# ## Homework 2
# 
# * Import Boston Dataset from sklearn dataset class.
# * Explore and analyse raw data.
# * Do preprocessing for regression.
# * Split your dataset into train and test test (0.7 for train and 0.3 for test).
# * Try Ridge and Lasso Regression models with at least 5 different alpha value for each.
# * Evaluate the results of all models and choose the best performing model.

# In[9]:


# Import boston dataset and convert it into pandas dataframe

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


Xb,yb = load_boston(return_X_y=True)
df_boston = pd.DataFrame(Xb,columns = load_boston().feature_names)
df_boston.head()


# In[10]:


# Check duplicate values and missing data

df_boston.isna().sum()


# In[11]:


# Visualize data for each feature (pairplot,distplot)


import seaborn as sns
sns.pairplot(df_boston)


# In[12]:


# Draw correlation matrix


df_boston.corr()


# In[15]:


# Drop correlated features (check correlation matrix)

new_df = df_boston.drop(["AGE","INDUS"],axis=1)


# In[18]:


# Handle outliers (you can use IsolationForest)

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_boston))
z


# In[19]:


len(np.where(z > 3)[0])


# In[20]:


outliers = list(set(np.where(z > 3)[0]))
new_df = df_boston.drop(outliers,axis = 0).reset_index(drop = False)
display(new_df)

y_new = yb[list(new_df["index"])]
len(y_new)


# In[21]:


# Normalize data

X_new = new_df.drop('index', axis = 1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_scaled = StandardScaler().fit_transform(X_new)
X_scaled


# In[30]:


# Split dataset into train and test set

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_new, test_size=0.3, random_state=42)
modelb = LinearRegression(normalize=False)

modelb.fit(X_train,y_train)

print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))


# In[53]:


#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

print(f'Regression model coef: {regression_model.coef_}')

print('*************************')

print("Simple Train: ", regression_model.score(X_train, y_train))
print("Simple Test: ", regression_model.score(X_test, y_test))


# In[39]:


# Import ridge and lasso models from sklearn

import pandas as pd
import numpy as np

#Import graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score


# In[40]:


# ridge regression

ridge_model = Ridge(alpha = 10)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')


# In[41]:


# lasso regression 

lasso_model = Lasso(alpha = 0.2)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')


# In[55]:


# Define 5 different alpha values for lasso and fit them. Print their R^2 sore on both
# train and test.

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************')

# 1
lasso_model = Lasso(alpha = 0.5)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************');


# 2
lasso_model = Lasso(alpha = 1)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************');

# 3
lasso_model = Lasso(alpha = 3)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************');

# 4
lasso_model = Lasso(alpha =0.1)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************');

# 5 
lasso_model = Lasso(alpha = 0.3)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')

#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************');


# In[58]:


# Define 5 different alpha values for Ridg and fit them. Print their R^2 sore on both
# train and test.

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')

# 1
ridge_model = Ridge(alpha = 5)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')

# 2
ridge_model = Ridge(alpha = 0.5)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')

# 3
ridge_model = Ridge(alpha = 0.1)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')

# 4
ridge_model = Ridge(alpha = 2)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')

# 5
ridge_model = Ridge(alpha = 2.5)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')

#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
print('*************************')


# ## Make comment about results. Print best models coefficient.
#     
# * Lineer regresyon modelinin test verisindeki performansı 0.6739782514175501'dir.
#  
# 
# * Ridge regresyon modeli için 5 farklı alpha değerinin test verisi üzerindeki performansını incelediğimde hepsinin benzer değerlerde olduğunu söyleyebilirim. Burada max değer alpha değeri = 0.1 için "0.673956615623817" olarak hesaplanmıştır.
# 
# 
# * Lasso regresyon modeli için 5 farklı alpha değerinin test verisi üzerindeki etkisini incelediğimde en iyi değerin "0.6583946315771074" olduğunu ve 0.1 alpha değerinde geldiğini söyleyebilirim. 
# 
# ###### Bu sonuçlar altında genel bir yorum yapmak istersek;
# 
# * Bu veri için şu an verdiğimiz değerlere göre en iyi performansı lineer regresyon modelimiz vermiştim. Ancak yine de lineer regresyon modelinin en iyisi olduğunu söyleyemeyiz. Farklı train-test verileri için daha düşük sonuçlar elde edebiliriz veya ridge ve lasso regresyon modellerinde farklı alpha değerleri için daha iyi sonuçlar alabiliriz. Sadece şimdilik belirlediğimiz durumlara göre en iyisinin lineer regresyon modeli olduğunu söyleyebiliriz. 
# 
