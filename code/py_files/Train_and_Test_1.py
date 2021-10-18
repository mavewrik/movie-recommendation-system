#!/usr/bin/env python
# coding: utf-8

# In[12]:


import ast
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from collections import Counter
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


actor_dir_genr_sim_matr = pickle.load(open("actor_dir_genr_sim_matr1",'rb'))


# In[45]:


overview_sim_matr = pickle.load(open("overview_sim_matr1",'rb'))


# In[46]:


actor_dir_genr_sim_matr.shape


# In[47]:


np.unique(actor_dir_genr_sim_matr[0])


# In[40]:


min(np.unique(actor_dir_genr_sim_matr[0]))


# In[48]:


Counter(actor_dir_genr_sim_matr[0]).most_common(70)


# In[41]:


Counter(overview_sim_matr[0]).most_common(70)


# In[4]:


all_doc = pickle.load(open("all_doc_1",'rb'))


# In[5]:


actor_dir_genr_key_docs, overview_docs, mapping = all_doc[0], all_doc[1], all_doc[2] 


# In[6]:


# mapping.index(832)
len(actor_dir_genr_key_docs)


# In[7]:


train_and_test_dict = pickle.load(open("train_and_test_dict_1",'rb'))


# In[8]:


train_dict, test_dict = train_and_test_dict[0], train_and_test_dict[1]


# In[9]:


len(train_dict)


# In[10]:


len(test_dict)


# In[13]:


def user_pred_errors(user_ratings, ratio, sim_matr_1, sim_matr_2, mapping):
    movie_ids = list(user_ratings.keys())
    np.random.shuffle(movie_ids)
    train_pts = len(movie_ids) - int(0.2 * len(movie_ids))
    
    train_ratings = np.array([user_ratings[mid] for mid in movie_ids[:train_pts]])
    train_ind = [mapping.index(mid) for mid in movie_ids[:train_pts]]
    
    test_ratings = np.array([user_ratings[mid] for mid in movie_ids[train_pts:]])
    test_ind = [mapping.index(mid) for mid in movie_ids[train_pts:]]
    
    user_mean = np.mean(train_ratings)
    errors = []
    for i in range(len(test_ratings)):
        pred = 0
        sim_sum = 0
        for j in range(len(train_ratings)):
            pred += ((sim_matr_1[test_ind[i]][train_ind[j]] * ratio) + (sim_matr_2[test_ind[i]][train_ind[j]] * (1-ratio))) * (train_ratings[j] - user_mean)
            sim_sum += ((sim_matr_1[test_ind[i]][train_ind[j]] * ratio) + (sim_matr_2[test_ind[i]][train_ind[j]] * (1-ratio)))
        pred = pred / sim_sum
        pred = pred + user_mean
        error = (pred - test_ratings[i])**2
        errors.append(error)
    return errors    


# In[14]:


def training(train_user_dict, sim_matr_1, sim_matr_2, mapping, start, incr): 
    a = []
    rmse = []
    print("Tested for alpha: ", end='')
    for alpha in np.arange(start, 1.0, incr):
        alpha = round(alpha,3)
        error = []
        for j in train_user_dict.keys():
            error += user_pred_errors(train_user_dict[j], alpha, sim_matr_1, sim_matr_2, mapping)
        
        error = np.array(error)
        mean_all = np.mean(error)
        sv = mean_all ** 0.5
        rmse.append(sv)
        a.append(alpha)
        print(alpha, end=' ')
    print()
    return a, rmse 


# In[19]:


ratios, rmse = training(train_dict, actor_dir_genr_sim_matr, overview_sim_matr, mapping, 0, 0.05)


# In[53]:


def get_plot(x, y, axis_names=['x-axis','y-axis'], title = ''):
    plt.figure(figsize=(12,5))
    plt.plot(x, y) 
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1]) 
#     plt.xlim(0.0,1)
    xi = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    plt.xticks(xi)
    plt.title(title,fontweight='bold')
    plt.grid()
    plt.show()


# In[54]:


get_plot(ratios, rmse, axis_names=['Alpha (ratio)','RMSE'], title = 'RMSE plot for different parameter values')


# In[28]:


rmse.index(min(rmse[1:]))


# In[27]:


rmse


# In[29]:


def testing(test_user_dict, sim_matr_1, sim_matr_2, mapping, alpha): 
    error = []
    for j in test_user_dict.keys():
        error += user_pred_errors(test_user_dict[j], alpha, sim_matr_1, sim_matr_2, mapping)
    error = np.array(error)
    mean_all = np.mean(error)
    rmse = mean_all ** 0.5
    return rmse


# In[31]:


test_rmse = testing(test_dict, actor_dir_genr_sim_matr, overview_sim_matr, mapping, 0.25)


# In[32]:


test_rmse


# In[ ]:




