#!/usr/bin/env python
# coding: utf-8

# ## Homework 3
# 
# * Generate dataset using `make_classification` function in the sklearn.datasets class. Generate 10000 samples with 8 features (X) with one label (y). Also, use following parameters    
#     * `n_informative` = 5
#     *  `class_sep` = 2
#     * `random_state` = 42
# * Explore and analyse raw data.
# * Do preprocessing for classification.
# * Split your dataset into train and test test (0.7 for train and 0.3 for test).
# * Try Decision Tree and XGBoost Algorithm with different hyperparameters. (Using GridSearchCV is a plus)
# * Evaluate your result on both train and test set. Analyse if there is any underfitting or overfitting problem. Make your comments.

# In[6]:


# Import necessary libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


# In[68]:


# Generate dataset using make_classification function in the sklearn. 
# Convert it into pandas dataframe.

features, output = make_classification(n_samples = 10000,
                                       n_features = 8,
                                       n_informative = 5,
                                       class_sep = 2,
                                       random_state = 42)


# In[125]:


print("Feature Matrix: ");
df = pd.DataFrame(features, columns=["Feature 1", "Feature 2", "Feature 3",
                                          "Feature 4", "Feature 5","Feature 6", "Feature 7", "Feature 8"]).head()
print(df)


# In[126]:


print()
print("Target Class: ");
print(pd.DataFrame(output, columns=["TargetClass"]).head())


# In[127]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(random_state = 42)


# In[128]:


model.fit(features, output)


# In[129]:


print("Classes: ", model.classes_)
print("Intercept: ",model.intercept_)
print("Coef: ",model.coef_)


# In[130]:


print("Probability estimates:","\n",model.predict_proba(features))


# In[131]:


model.predict(features)
print("Actual (class) predictions:","\n",model.predict(features))


# In[139]:


# Check duplicate values and missing data.
df.isna().sum()


# In[143]:


df.duplicated().sum()


# In[147]:


# Visualize data for each feature (pairplot,distplot).

sns.pairplot(df)


# In[148]:


sns.distplot(df)


# In[133]:


# Draw correlation matrix.

df.corr()


# In[134]:


# Handle outliers (you can use IsolationForest, Z-score, IQR)

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df))
z


# In[149]:


outliers = list(set(np.where(z > 3)[0]))

len(outliers)

### outlier değer yoktur


# In[154]:


# Split dataset into train and test set


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(features,output, test_size=0.3, random_state=42)

models = LogisticRegression(random_state=42, n_jobs=-1)
cv = cross_validate(models,X_train,y_train, cv = 3, n_jobs=-1, return_estimator=True)

print("Mean training accuracy: {}".format(np.mean(cv['test_score'])))
print("Test accuracy: {}".format(cv["estimator"][0].score(X_test,y_test)))


# In[185]:


# Import Decision Tree, define different hyperparamters and tune the algorithm.

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))

print("Accuracy of test:",clf.score(X_test,y_test))


from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
pred = clf.predict(X_test)
print(classification_report(y_test,pred))


# In[184]:


# Visualize feature importances.



# In[93]:


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

### Confusion Matrix
y_pred = model.predict(features)
confusion_matrix(output, y_pred)


# In[94]:


### accuracy, recall, precision and f1 score.
y_pred = model.predict(features)
print(classification_report(output,y_pred))


# In[187]:


# Import XGBoostClassifier, define different hyperparamters and tune the algorithm.

get_ipython().system('pip install xgboost')
import xgboost as xgb

dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)


# In[188]:


param = {'max_depth':3, 
         'eta':1, 
         'objective':'multi:softprob', 
         'num_class':3}

num_round = 5
model = xgb.train(param, dmatrix_train, num_round)


# In[191]:



preds = model.predict(dmatrix_test)
preds[:10]


# In[192]:


best_preds = np.asarray([np.argmax(line) for line in preds])
best_preds


# In[194]:


# Visualize feature importances.


# In[195]:


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))


# In[ ]:


# Evaluate your result and select best performing algorithm for our case.

