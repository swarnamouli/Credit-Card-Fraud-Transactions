#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Seaborn: {}'.format(seaborn.__version__))


# In[2]:


#import the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load the dataset
data=pd.read_csv('creditcard.csv')
print(data.columns)


# In[4]:


print(data.shape)


# In[5]:


#create a sample/fraction of the dataset

data=data.sample(frac=0.1, random_state = 1)
print(data.shape)


# In[6]:


#plot a histogram of each parameter
data.hist(figsize=(20,20))
plt.show()


# In[ ]:





# In[7]:


# Determine number of fraud cases

fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outlier_fraction= len(fraud) / float(len(valid))
print(outlier_fraction)
print('Fraud Cases: {}'. format(len(fraud)))
print('Valid cases: {}'.format(len(valid)))


# In[11]:


#Correlation Matrix

corrmat =data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True, annot=False)
plt.show()


# In[15]:


# get all columns from dataframe
columns=data.columns.tolist()

#filter the columns to remove data which is not wanted

columns= [c for c in columns if c not in ["Class"]]

#store the variables predicting on 

target="Class"
X=data[columns]
Y=data[target]

#Print the shapes of X and Y

print(X.shape)
print(Y.shape)


# In[16]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state=1

#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination=outlier_fraction,
                                       random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors=20,
    contamination=outlier_fraction)
}


# In[21]:


#Fit the model

n_outliers=len(fraud)

#for loop through the two different classifiers defined above
for i, (clf_name,clf) in enumerate(classifiers.items()):
    
    #fir the data and log outliers
    if clf_name =="Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
        #Reshape the prediction to : 0 for valid and 1 for fraud
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        
        
        n_errors=(y_pred != Y).sum()
        
        #run classification metrics
        
        print('{}: {}'.format(clf_name, n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y,y_pred))
        

