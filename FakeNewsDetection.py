#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas sklearn


# In[2]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[9]:


#Read the data  into a DataFrame
df=pd.read_csv('/Users/jalilkhan/Downloads/news.csv')

#Get shape and head
df.shape
df.head()


# In[10]:


#DataFlair - Get the labels

labels=df.label
labels.head()


# In[11]:


#DataFlair - Split the dataset

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# ## initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 -- terms with a higher document frequency will be discarded.
# ## Stop words are to be filtered out before processing the natural language data. 
# ## A TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.

# In[12]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[13]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[14]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# ## So with this model, we have 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives.

#  what we did here is we took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.82% in magnitude.

# In[ ]:




