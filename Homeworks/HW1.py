#!/usr/bin/env python
# coding: utf-8

# ## Homework 1
# 
# 1) How would you define Machine Learning?  
# 2) What are the differences between Supervised and Unsupervised Learning? Specify example 3 algorithms   for each of these.  
# 3) What are the test and validation set, and why would you want to use them?  
# 4) What are the main preprocessing steps? Explain them in detail. Why we need to prepare our data?  
# 5) How you can explore countionus and discrete variables?  
# 6) Analyse the plot given below. (What is the plot and variable type, check the distribution and make comment about how you can preproccess it.) 

# ### Answer1) 
# Machine learning is a science that enables machines to learn like a human by analyzing data with certain algorithms. It is a subset of artificial intelligence. As the amount of data and trials increases, the results of machine learning become more accurate.

# ### Answer2)
# 1. Supervised learning has input and output variables. In unsupervised learning, there is only an input variable. 
# 2. The supervised learning model receives direct feedback to check whether it predicts the correct output. Unsupervised learning does not receive any feedback from the model.
# 3. The purpose of supervised learning is to train the model so that it can predict output when new data is given. The purpose of unsupervised learning is to find hidden patterns and useful insights from the unknown data set.
# 

# ### Answer3)
#    The trainer is used to match models; The verification set is used to estimate the estimation error for model selection;
# The test set is used to evaluate the generalization error of the final model selected. Ideally, the test set is
# it should be kept and revealed only at the end of the data analysis. The test set is used only to test and tell the accuracy of the system. 
# 
# 
# *
# 
# 
#    It is often used for parameter selection and to prevent overfitting. If your model is nonlinear and trained on only one training set, it is very likely to achieve 100% accuracy and overfitting, so it gets very low performance on the test set. Therefore, a verification set independent of the training set is used for parameter selection.

# ### Answer.4)
# 
# 
# 1. Selecting Data
# Our goal in this step is to select a subset from all available data we will be working with. While it is more tempting to select all the data, this is not always true.
# First of all, we should consider what data we need to solve the problem we are working on. We should make some inferences and make variations on the data we need.
# 
# 
# 2) Preparing the Data
# Data preparation, where we load our data into a suitable place and prepare it for use in our machine learning training. This is also a good time to do any pertinent visualizations of your data, to help you see if there are any relevant relationships between different variables you can take advantage of, as well as show you if there are any data imbalances.
# 
# ---Exploratory Data Analysis (EDA); Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# 
# 
# ###### Pre-Processing
# Duplicate Values
# In most cases, the duplicates are removed so as to not give that particular data object an advantage or bias, when running machine learning algorithms.
# 
# ###### Imbalanced Data
# An Imbalanced dataset is one where the number of instances of a class(es) are significantly higher than another class(es), thus leading to an imbalance and creating rarer class(es).
# 
# ###### Missing Values
# Eleminate missing values
# Filling with mean or median
# 
# ###### Feature Scaling
# Standardization
# Normalization
# 
# ###### Bucketing (Binning)
# Data binning, bucketing is a data pre-processing method used to minimize the effects of small observation errors (noisy data). The original data values are divided into small intervals known as bins and then they are replaced by a general value calculated for that bin.
# 
# ###### Feature Extraction
# Principle Components Analysis (PCA)
# Independent Component Analysis (ICA)
# Linear Discriminant Analysis (LDA)
# t-distributed Stochastic Neighbor Embedding (t-SNE)
# 
# ###### Feature Encoding
# Feature encoding is basically performing transformations on the data such that it can be easily accepted as input for machine learning algorithms while still retaining its original meaning.
# 
# Nominal : Any one-to-one mapping can be done which retains the meaning. For instance, a permutation of values like in One-Hot Encoding.
# Ordinal : An order-preserving change of values. The notion of small, medium and large can be represented equally well with the help of a new function. For example, we can encode this S, M and L sizes into {0, 1, 2} or maybe {1, 2, 3}.
# 
# 
# ##### Train / Validation / Test Split
# But before we start deciding the algorithm which should be used, it is always advised to split the dataset into 2 or sometimes 3 parts. Machine Learning algorithms, or any algorithm for that matter, has to be first trained on the data distribution available and then validated and tested, before it can be deployed to deal with real-world data.
# 
# --- 60 / 20 / 20
# 
# --- 70 / 30
# 
# ##### Cross Validation
# Cross-validation is a statistical resampling method used to evaluate the performance of the machine learning model on data it cannot see as objectively and accurately as possible.

# ### Answer.5)
# While the field of a continuous variable consists of all real values in a certain range, the area of other variables can be counted at most. However, continuous variables are defined as numbers, but as continuous measurements.
# 
# **Some examples of discrete data that can be collected:
# 
# -- Number of customers purchasing different products
# 
# -- Number of computers in each department
# 
# -- The number of products you buy from the market each week
# 
# **Some examples of continuous data include:
# 
# -- The weight of newborns
# 
# -- Daily wind speed
# 
# --Temperature of the freezer

# 
# ### Answer.6)
# 
# (note: I accidentally lost the chart and could not get it back.)
# 
# ** Histogram-Frequency table
# 
# ** The variable is continuous
# 
# ** It may be better to categorize the variable and plot it. For example, we can collect those between 0-1 and those between 1-2 and 2-3 under a single variable.

# In[ ]:




