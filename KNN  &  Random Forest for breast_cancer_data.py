#!/usr/bin/env python
# coding: utf-8

# ## KNN
# ### Simplicity:
# Easy to understand and implement, making it beginner-friendly.
# ### No Training Phase:
# Functions as a "lazy learner," storing the training data and making predictions on-the-fly.
# ### Adaptability:
# Easily accommodates new data without retraining.
# ### Versatility: 
# Suitable for both classification and regression tasks.
# ### Non-parametric: 
# Makes no assumptions about data distribution, allowing for broad applicability.
# ### Effective for Non-linear Data:
# Captures complex relationships and non-linear decision boundaries.
# ### Customizable Distance Metrics: 
# Users can choose different distance measures based on specific needs.
# 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("D:\\sandip sir 3rd sem lab\\breast_cancer_dataset.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


X=df.iloc[:,0:30]
X.head()


# In[6]:


y=df.iloc[:,-1]
y.head()


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,0:30],df.iloc[:,-1],test_size=0.2,random_state=2)


# In[8]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[10]:


X_train


# In[11]:


X_train.shape


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


# In[13]:


knn.fit(X_train,y_train)


# In[14]:


from sklearn.metrics import accuracy_score ,classification_report
y_pred=knn.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# Precision is 0.98 it means 98% of instances predicted as class '0' were correct.
# for class 1 Precision is 1.00 indicating perfect precision all instances predicted as class '1' were correct.
# 
# Recall also known as sensitivity or True Positive Rate measures the ratio of true positive predictions to the total actual positives.here 1 indicates 0 correctly indentified.
# 0.99 indicating that 99% of actual instances of class '1' were correctly indentified.
# F1 scoring performing the 0.99 suggesting that the model performs well in balancing precision and recall.
# 
# these metrics  classification model performs exceptionally well across both classes with high precision, recall, and F1-scores close to perfect (1.00). The accuracy rate further confirms that it effectively distinguishes between classes with minimal errors. This analysis suggests that model is robust and reliable for making predictions in this context.
# 

# In[15]:


# Initialize scores list
scores = []

for i in range (1,16):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))


# In[16]:


import matplotlib.pyplot as plt
plt.plot(range(1,16),scores)
plt.show()


# ## Random forest 
# 1. High accuracy
# Random Forest generally provides high accuracy in predictions due to its ensemble nature. 
# 
#  it reduces the risk of overfitting that is common with single decision trees.
#  
# 2. Robustness to Noise and Outliers
# predictions are based on the majority vote from multiple trees, the influence of any single noisy data point is minimized.
# 
# 3. Handles High-Dimensional Data
# it is especially beneficial in fields like genomics or image processing where high-dimensional data is common. It automatically performs feature selection and can indicate the importance of different features in making predictions.
# 
# 4. Non-Parametric Nature
# Random Forest does not assume any specific distribution for the data.
# 
# 5. Easy to Use and Interpret
# it provides insights into feature importance, helping users understand which variables significantly impact predictions.
# 
# 6. Parallel Processing Capability
# This property enhances computational efficiency, especially when dealing with large datasets.
# 
# 7. Versatile Applications
# Random Forest is applicable across various domains, including finance (for credit scoring and fraud detection), healthcare (for disease diagnosis), and e-commerce (for customer behavior prediction). Its versatility makes it a go-to choice for many machine learning tasks.
# 
# 8. Good Performance with Minimal Data Preprocessing
# Random Forest can handle missing values and does not require extensive preprocessing steps like normalization or one-hot encoding for categorical variables, simplifying the data preparation process
# 

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[18]:


df=pd.read_csv("D:\\sandip sir 3rd sem lab\\breast_cancer_dataset.csv")
df.head()


# In[19]:


X=df.iloc[:,0:30]
X


# In[20]:


y=df.iloc[:,-1]
y


# In[21]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[23]:


# Train the model
rf_classifier.fit(X_train, y_train)


# In[24]:


# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)


# In[25]:


# Evaluate the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


#  its accuracy is 0.9649 its a better performence.
