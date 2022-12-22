#!/usr/bin/env python
# coding: utf-8

# # **Book Rental Recommendation**

# Before reading data from a csv file, you need to download the “BX-Book-Ratings.csv”, “BX-Books.csv”, “BX-Users.csv”, and “Recommend.csv” datasets from the resource section and upload them into the Lab.
# We will use the Up arrow icon, which is shown on the left side under the View icon. Click on the Up arrow icon and upload the file 
# from wherever it was downloaded into your system.
# 
# The objective is to recommend books to a user based on their purchase history and the behavior of other users.
# 

# ### **Dataset Description**
# 
# * BX-Users: It contains the information of users.
#   * user_id - These have been anonymized and mapped to integers
#   * Location - Demographic data is provided
#   * Age - Demographic data is provided
# * If available. Otherwise, these fields contain NULL-values.
# 
# * BX-Books: 
#   * isbn - Books are identified by their respective ISBNs. Invalid ISBNs have already been removed from the dataset.
#   * book_title
#   * book_author
#   * year_of_publication
#   * publisher
# * BX-Book-Ratings: Contains the book rating information. 
#   * user_id
#   * isbn
#   * rating - Ratings (`Book-Rating`) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.
# 
# 
# 

# ## Getting Started

# In[1]:


#Import required libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# #### Read the data using pandas in DataFrame df_user

# To map byte values directly to the first 256 Unicode code points, use the "Latin-1" encoding. This is the closest equivalent Python 3 offers to the permissive Python 2 text handling model.

# In[2]:


#Reading the data
df_user = pd.read_csv('BX-Users.csv',encoding='latin-1')


# #### Preview the information of first 5 rows of dataset.

# In[3]:


df_user.head()


# #### Checking for Null Values

# In[4]:


df_user.isnull().any()


# #### Dropping the Null Values

# In[5]:


df_user1=df_user.dropna()


# In[6]:


df_user1.isnull().any()


# #### Read the books Data and explore

# In[7]:


df_books = pd.read_csv('BX-Books.csv', encoding='latin-1')


# #### Preview the information of first 5 rows of dataset.

# In[8]:


df_books.head()


# #### Reading the data where ratings are given
# You will read only first 10000 rows otherwise, Out Of Memory  error can occur.

# In[9]:


df = pd.read_csv('BX-Book-Ratings.csv',encoding='latin-1',nrows=10000)


# #### Preview the information of first 5 rows of dataset.

# In[10]:


df.head()


# #### Using **'describe()'** function 
# It is used to view some basic statistical details like percentile, mean, std.

# In[11]:


df.describe()


# #### Merge the dataframes. 
# For all practical purposes, User Master Data is not required. So, ignore dataframe df_user

# In[12]:


df = pd.merge(df,df_books,on='isbn')
df.head()


# #### Checking for unique users and books
# 

# Here we are using **'nunique()'** function that returns the Series with the number of distinct observations over the requested axis.

# In[13]:


#Code for checking number of unique users and books. 
n_users = df.user_id.nunique()
n_books = df.isbn.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Books: '+str(n_books))


# #### Convert ISBN variable to numeric type in order 

# In[14]:


#Convert and print length of isbn list
isbn_list = df.isbn.unique()
print(" Length of isbn List:", len(isbn_list))
def get_isbn_numeric_id(isbn):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(isbn_list==isbn)
    return itemindex[0][0]


# 
# 
# ```
# # This is formatted as code
# ```
# 
# #### Convert user_id variable to numeric type in order

# In[15]:


#Convert and print length of user_id list
userid_list = df.user_id.unique()
print(" Length of user_id List:", len(userid_list))
def get_user_id_numeric_id(user_id):
    #print ("  isbn is:" , isbn)
    itemindex = np.where(userid_list==user_id)
    return itemindex[0][0]


# #### Convert both user_id and isbn to ordered list i.e. from 0...n-1

# In[16]:


df['user_id_order'] = df['user_id'].apply(get_user_id_numeric_id)


# In[17]:


df['isbn_id'] = df['isbn'].apply(get_isbn_numeric_id)
df.head()


# #### Re-index columns to build matrix

# In[18]:


#Reindexing the columns
new_col_order = ['user_id_order', 'isbn_id', 'rating', 'book_title', 'book_author','year_of_publication','publisher','isbn','user_id']
df = df.reindex(columns= new_col_order)
df.head()


# ## Train Test Split
# 
# Recommendation Systems  are difficult to evaluate, but you will still learn how to evaluate them. In order to do this, you'll split your data into two sets. However, you won't do your classic X_train,X_test,y_train,y_test split. Instead, you can actually just segement the data into two sets of data:

# #### Importing train_test_split model

# In[19]:


#Importing train_test_split model for splittig the data into train and test set
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.30)


# ## Approach: You Will Use Memory-Based Collaborative Filtering
# 
# Memory-Based Collaborative Filtering approaches can be divided into two main sections: **user-item filtering** and **item-item filtering**. 
# 
# A *user-item filtering* will take a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked. 
# 
# In contrast, *item-item filtering* will take an item, find users who liked that item, and find other items that those users or similar users also liked. It takes items as input and outputs other items as recommendations. 
# 
# * *Item-Item Collaborative Filtering*: “Users who liked this item also liked …”
# * *User-Item Collaborative Filtering*: “Users who are similar to you also liked …”

# In both cases, you create a user-book matrix which is built from the entire dataset.
# 
# Since you have split the data into testing and training, you will need to create two ``[828 x 8051]`` matrices (all users by all books). This is going to be  a very large matrix
# 
# The training matrix contains 70% of the ratings and the testing matrix contains 30% of the ratings.  

# #### Create two user-book matrix for training and testing
# 
# > Indented block
# 
# 

# In[20]:


#Create user-book matrix for training 
train_data_matrix = np.zeros((n_users, n_books))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  
    
#Create user-book matrix for testing
test_data_matrix = np.zeros((n_users, n_books))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


# #### Import Pairwise Model 

# You can use the [pairwise_distances](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html) function from sklearn to calculate the cosine similarity. Note, the output will range from 0 to 1 since the ratings are all positive.

# In[21]:


#Importing pairwise_distances function
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[22]:


user_similarity


# #### Make predictions

# In[23]:


#Defining custom function to make predictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[24]:


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# ### Evaluation
# There are many evaluation metrics, but one of the most popular metric used to evaluate accuracy of predicted ratings is *Root Mean Squared Error (RMSE)*.

# Since, you only want to consider predicted ratings that are in the test dataset, you filter out all other elements in the prediction matrix with: `prediction[ground_truth.nonzero()]`. 

# In[25]:


#Importing RMSE function 
from sklearn.metrics import mean_squared_error
from math import sqrt

#Defining custom function to filter out elements with ground_truth.nonzero
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# #### Printing RMSE value for user based and item based collaborative filtering

# In[26]:


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# ### Both the approach yield almost same result
