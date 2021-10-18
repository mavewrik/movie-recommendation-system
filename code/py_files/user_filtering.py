#!/usr/bin/env python
# coding: utf-8

# In[2]:


import  glob
import csv
import pickle
# from sklearn import 


# In[23]:


file = []


# In[16]:


with open("Dataset\\ratings.csv",'r') as f:
    data= csv.DictReader(f)
    for row in data:
        file.append(row)


# In[17]:


file[0]


# In[19]:


k = ['userId', 'movieId', 'rating']


# In[33]:


small_dataset_of_rating = []


# In[20]:


for i in range(len(file)):
    temp = []
    for j in k:
        temp.append(file[i][j])
    small_dataset_of_rating.append(temp)


# In[28]:


rating_dict = {}
for i in range(len(small_dataset_of_rating)):
    movie = int(small_dataset_of_rating[i][1])
    user = int(small_dataset_of_rating[i][0])
    rating = float(small_dataset_of_rating[i][2])
    if rating_dict.get(user) is None:
        rating_dict[user] = {movie: rating}
    else:
        rating_dict[user][movie] = rating


# In[34]:


with open("rating_dict", 'wb') as fp:
    pickle.dump(rating_dict, fp)


# In[4]:


len(rating_dict)


# In[3]:


rating_dict = pickle.load(open("rating_dict",'rb'))


# In[33]:


rating_small_dict = dict()


# In[34]:


count = 0
for user in list(rating_dict.keys()):
    if len(rating_dict[user]) > 50:
        count += 1
        rating_small_dict[user] = rating_dict[user]
    if count == 500:
        break


# In[35]:


len(rating_small_dict)


# In[36]:


with open("rating_small_dict", 'wb') as fp:
    pickle.dump(rating_small_dict, fp)


# In[39]:


x = []
for usr in rating_dict.keys():
    x.append(len(rating_dict[usr]))
min(x)


# In[ ]:




