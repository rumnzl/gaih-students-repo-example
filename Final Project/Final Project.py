#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# In this project, our aim is to building a model for predicting dimond prices. Our label (output) will be `price` column. **Do not forget, this is a Classification problem!**
# 
# ## Content
# carat: weight of the diamond (0.2--5.01)
# 
# cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
# color: diamond colour, from J (worst) to D (best)
# 
# clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
# x: length in mm (0--10.74)
# 
# y: width in mm (0--58.9)
# 
# z: depth in mm (0--31.8)
# 
# depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# 
# table: width of top of diamond relative to widest point (43--95)
# 
# ## Steps
# - Read the `diamonds.csv` file and describe it.
# - Make at least 4 different analysis on Exploratory Data Analysis section.
# - Pre-process the dataset to get ready for ML application. (Check missing data and handle them, can we need to do scaling or feature extraction etc.)
# - Define appropriate evaluation metric for our case (classification). *Hint: Is there any imbalanced problem in the label column?*
# - Split the dataset into train and test set. (Consider the imbalanced problem if is there any). Check the distribution of labels in the subsets (train and test).
# - Train and evaluate Decision Trees and at least 2 different appropriate algorithm which you can choose from scikit-learn library.
# - Is there any overfitting and underfitting? Interpret your results and try to overcome if there is any problem in a new section.
# - Create confusion metrics for each algorithm and display Accuracy, Recall, Precision and F1-Score values.
# - Analyse and compare results of 3 algorithms.
# - Select best performing model based on evaluation metric you chose on test dataset.
# 
# 
# Good luck :)

# <h2>Your Name</h2>

# # Data

# In[4]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# Read csv

df = pd.read_csv("diamonds.csv")
df


# In[6]:


# Describe our data for each feature and use .info() for get information about our dataset
# Analyse missing values

df.info()

df.isna().sum()


# # Exploratory Data Analysis

# In[7]:


# Our label Distribution (countplot)

sns.countplot(x='cut', data=df)


# In[8]:


# Example EDA (distplot)

sns.distplot(df.table)


# # Preprocessing
# 
# - Are there any duplicated values?
# - Do we need to do feature scaling?
# - Do we need to generate new features?
# - Split dataset into train and test sets. (0.7/0.3)

# In[9]:


df.duplicated()

df.duplicated().sum()

## duplicated value yok. 


# In[10]:


sns.distplot(df.price) 

## evet scaling yapmamız geerek verimiz normal dağılmıyor.


# ### train-test split

# In[11]:


dependent_variable = df["price"]
predictors = df.drop("price", axis = 1)
display(dependent_variable.describe())

display(predictors.head(5))


# # ML Application
# 
# - Define models.
# - Fit models.
# - Evaluate models for both train and test dataset.
# - Generate Confusion Matrix and scores of Accuracy, Recall, Precision and F1-Score.
# - Analyse occurrence of overfitting and underfitting. If there is any of them, try to overcome it within a different section.

# In[12]:


def OrdererdListEncoder(ord_list):
    """
    ord_list: a python list with predetermined order
    return a dictionary that maps values in ord_list to a ranking
    """
    return {ord_list[i]: len(ord_list) - i for i in range(len(ord_list))}


# In[13]:


# Encode Carat, Color and Cut
cut_rank = OrdererdListEncoder(['Ideal','Premium','Very Good', 'Good', 'Fair'])
color_rank = OrdererdListEncoder(list('DEFGHIJ'))
clarity_rank = OrdererdListEncoder(['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])

df['cut'] = df['cut'].apply(lambda x: cut_rank[x])
df['color'] = df['color'].apply(lambda x: color_rank[x])
df['clarity'] = df['clarity'].apply(lambda x: clarity_rank[x])

df.head()


# In[26]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=regressor.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))


# ### train-test split

# In[21]:





# In[22]:





# In[23]:





# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:





# # Evaluation
# 
# - Select the best performing model and write your comments about why choose this model.
# - Analyse results and make comment about how you can improve model.
