#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import math
import pickle
import numpy as np
import copy
from collections import Counter
import matplotlib.pyplot as plt
from prettytable import PrettyTable
# from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
sys.path.append('C:\Python36\Lib\site-packages')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


mapping = pickle.load(open('./mapping', 'rb'))
train_dict, test_dict = pickle.load(open('./train_and_test_dict', 'rb'))


# <h3> Similarity matrix generating function</h3>

# In[3]:


def get_sim_matr(tf_idf_vecs):
    sim_matr = cosine_similarity(tf_idf_vecs, tf_idf_vecs)
    min_val = np.min(sim_matr)
    if min_val < 0:
        max_val = np.max(sim_matr)
        sim_matr = (sim_matr - min_val)/(max_val - min_val)
    smoothing_val  = np.unique(sim_matr)[1]
    sim_matr += smoothing_val
    sim_matr = sim_matr / np.max(sim_matr, axis=1)
    return sim_matr


# <h3>For F2 feature combinations : overview (without pca)</h3>

# In[ ]:


f2_tf_idf_vecs = pickle.load(open('./f2_tf_idf_vecs', 'rb'))


# In[4]:


f2_sim_mat = get_sim_matr(f2_tf_idf_vecs)


# In[5]:


f2_dim = f2_tf_idf_vecs.shape[1]


# In[6]:


np.min(f2_sim_mat)


# <h3> Train and Test functions </h3>

# In[7]:


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
            sim_val = ((sim_matr_1[test_ind[i]][train_ind[j]] * ratio) + (sim_matr_2[test_ind[i]][train_ind[j]] * (1-ratio)))
            pred += sim_val * (train_ratings[j] - user_mean)
            sim_sum += sim_val
        pred = pred / sim_sum
        pred = pred + user_mean
        error = (pred - test_ratings[i])**2
        if error > 25:
            print(error, ratio, pred, test_ratings[i])
        errors.append(error)
    return errors


# In[8]:


def training(train_user_dict, sim_matr_1, sim_matr_2, mapping, start, incr): 
    a = []
    rmse = []
    print("Tested for alpha: ", end='')
    for alpha in np.arange(start, 1.01, incr):
        alpha = round(alpha,2)
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


# In[9]:


def testing(test_user_dict, sim_matr_1, sim_matr_2, mapping, alpha): 
    error = []
    for j in test_user_dict.keys():
        error += user_pred_errors(test_user_dict[j], alpha, sim_matr_1, sim_matr_2, mapping)
    error = np.array(error)
    mean_all = np.mean(error)
    rmse = mean_all ** 0.5
    return rmse


# In[10]:


def f1_train_test(f1_sim_matr, f2_sim_matr, mapping, train_dict, test_dict):
    alphas, rmse_list = training(train_dict, f1_sim_matr, f2_sim_matr, mapping, 0, 0.05)
    
    get_plot(alphas, rmse_list, axis_names=['Alpha (ratio)','RMSE'], title = 'RMSE plot for different parameter(alpha) values')
    min_ind = rmse_list.index(min(rmse_list))
    best_alpha = alphas[min_ind]
    train_rmse = round(rmse_list[min_ind], 4)
    
    test_rmse = round(testing(test_dict, f1_sim_matr, f2_sim_matr, mapping, best_alpha), 4)    
    print("Results:-  Alpha: ", best_alpha,"   Train RMSE: ", train_rmse, "   Test RMSE: ", test_rmse)
    return test_rmse, train_rmse, best_alpha


# ##### PCA functions, table and plot functions

# In[11]:


# f1_13_tf_idf_vecs_copy = copy.deepcopy(f1_13_tf_idf_vecs)


# In[11]:


def get_pca(tf_idf_vecs):
    pca_var = PCA(n_components=0.99).fit(tf_idf_vecs)
    evecs_all = pca_var.components_
    variance_ratios = pca_var.explained_variance_ratio_
    return evecs_all, variance_ratios


# In[19]:


def get_test_data_pca(eig_energy_all, evecs_all, f1_13_vecs, f2_sim_mat, mapping, train_dict, test_dict, comb):
    eig_energy = np.arange(0.50, 1.0, 0.05)
    eig_energy = np.append(eig_energy, [0.99])
    best_test_rmse = math.inf
    train_rmse_list, test_rmse_list, alpha_list, dim_list = [], [], [], []
    
    for lambda_val in eig_energy:
        count = 0
        lambda_val_first = eig_energy_all[0]
        while lambda_val_first < lambda_val:
            count += 1
            lambda_val_first += eig_energy_all[count]
        lambda_val_first -= eig_energy_all[count]
        eig_vecs = evecs_all[:count]
        pca_transformed = np.matmul(f1_13_vecs, eig_vecs.T)
        pca_trans_sim_mat = get_sim_matr(pca_transformed)
#         print('sim_min', np.min(pca_trans_sim_mat), 'sim_max', np.max(pca_trans_sim_mat))
        print("Eigen energy: %8.5f" %(lambda_val_first))
        test_rmse, train_rmse, alpha = f1_train_test(pca_trans_sim_mat, f2_sim_mat, mapping, train_dict, test_dict)
        
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_pca_transformed = copy.deepcopy(pca_transformed)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        alpha_list.append(alpha)
        dim_list.append(pca_transformed.shape[1])
        
        print("---------------------------------------------------------------------")
    
    with open("Vec_space_after_pca\\f1_phase2_comb_"+str(comb), 'wb') as bpt:
        pickle.dump(best_pca_transformed, bpt)
    
    return train_rmse_list, test_rmse_list, alpha_list, dim_list, eig_energy


# In[52]:


def get_plot(x, y, axis_names=['x-axis','y-axis'], title = '', dim_list = []):
    plt.figure(figsize=(12,5))
    plt.plot(x, y) 
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.xticks(x)
    plt.title(title,fontweight='bold')
    if dim_list:
        for i, txt in enumerate(dim_list):
            plt.annotate(txt, (x[i], y[i]))
    plt.grid()
    plt.show()


# In[14]:


def get_table_per_feature_comb(var_ratio_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list):
    x = PrettyTable()
    x.field_names = ["Index", "Variance ratio", "F1 dim", "F2 dim", "alpha", "Train RMSE", "Test RMSE"]
    for i in range(len(var_ratio_list)):
        x.add_row([str(i+1), round(var_ratio_list[i], 2), f1_dim_list[i], f2_dim, alpha_list[i], train_rmse_list[i], test_rmse_list[i]])
    print(x)


# <h3> For F1 feature combination: countries</h3>

# In[20]:


comb = 5


# In[21]:


f_tf_idf_vecs = pickle.load(open('Vec_space_without_pca\\f1_5_tf_idf_vecs', 'rb'))


# In[22]:


evecs_all, variance_ratios = get_pca(f_tf_idf_vecs)


# In[23]:


train_rmse_list, test_rmse_list, alpha_list, f1_dim_list, threshold_var_list = get_test_data_pca(variance_ratios,                                                                                            evecs_all,                                                                                            f_tf_idf_vecs,                                                                                            f2_sim_mat,                                                                                            mapping,                                                                                            train_dict,                                                                                            test_dict, comb)


# In[24]:


get_table_per_feature_comb(threshold_var_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list)


# In[25]:


index = test_rmse_list.index(min(test_rmse_list))
res1 = [threshold_var_list[index], f1_dim_list[index], alpha_list[index], test_rmse_list[index]]

with open("Results\\res_"+str(comb), 'wb') as bpt:
        pickle.dump(res1, bpt)


# <h3> For F1 feature combination: countries, genres</h3>

# In[26]:


comb = 9


# In[27]:


f_tf_idf_vecs = pickle.load(open('Vec_space_without_pca\\f1_' + str(comb) + '_tf_idf_vecs', 'rb'))


# In[28]:


evecs_all, variance_ratios = get_pca(f_tf_idf_vecs)


# In[29]:


train_rmse_list, test_rmse_list, alpha_list, f1_dim_list, threshold_var_list = get_test_data_pca(variance_ratios,                                                                                            evecs_all,                                                                                            f_tf_idf_vecs,                                                                                            f2_sim_mat,                                                                                            mapping,                                                                                            train_dict,                                                                                            test_dict, comb)


# In[30]:


get_table_per_feature_comb(threshold_var_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list)


# In[31]:


index = test_rmse_list.index(min(test_rmse_list))
res1 = [threshold_var_list[index], f1_dim_list[index], alpha_list[index], test_rmse_list[index]]

with open("Results\\res_"+str(comb), 'wb') as bpt:
        pickle.dump(res1, bpt)


# <h3> For F1 feature combination:  countries, genres, actors</h3>

# In[32]:


comb = 10


# In[33]:


f_tf_idf_vecs = pickle.load(open('Vec_space_without_pca\\f1_' + str(comb) + '_tf_idf_vecs', 'rb'))


# In[34]:


evecs_all, variance_ratios = get_pca(f_tf_idf_vecs)


# In[35]:


train_rmse_list, test_rmse_list, alpha_list, f1_dim_list, threshold_var_list = get_test_data_pca(variance_ratios,                                                                                            evecs_all,                                                                                            f_tf_idf_vecs,                                                                                            f2_sim_mat,                                                                                            mapping,                                                                                            train_dict,                                                                                            test_dict, comb)


# In[36]:


get_table_per_feature_comb(threshold_var_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list)


# In[37]:


index = test_rmse_list.index(min(test_rmse_list))
res1 = [threshold_var_list[index], f1_dim_list[index], alpha_list[index], test_rmse_list[index]]

with open("Results\\res_"+str(comb), 'wb') as bpt:
        pickle.dump(res1, bpt)


# <h3> For F1 feature combination: countries, genres, actors, directors</h3>

# In[38]:


comb = 13


# In[39]:


f_tf_idf_vecs = pickle.load(open('Vec_space_without_pca\\f1_' + str(comb) + '_tf_idf_vecs', 'rb'))


# In[40]:


evecs_all, variance_ratios = get_pca(f_tf_idf_vecs)


# In[41]:


train_rmse_list, test_rmse_list, alpha_list, f1_dim_list, threshold_var_list = get_test_data_pca(variance_ratios,                                                                                            evecs_all,                                                                                            f_tf_idf_vecs,                                                                                            f2_sim_mat,                                                                                            mapping,                                                                                            train_dict,                                                                                            test_dict, comb)


# In[42]:


get_table_per_feature_comb(threshold_var_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list)


# In[43]:


index = test_rmse_list.index(min(test_rmse_list))
res1 = [threshold_var_list[index], f1_dim_list[index], alpha_list[index], test_rmse_list[index]]

with open("Results\\res_"+str(comb), 'wb') as bpt:
        pickle.dump(res1, bpt)


# <h3> For F1 feature combination: countries, genres, actors, keyterms </h3>

# In[44]:


comb = 14


# In[45]:


f_tf_idf_vecs = pickle.load(open('Vec_space_without_pca\\f1_' + str(comb) + '_tf_idf_vecs', 'rb'))


# In[46]:


evecs_all, variance_ratios = get_pca(f_tf_idf_vecs)


# In[47]:


train_rmse_list, test_rmse_list, alpha_list, f1_dim_list, threshold_var_list = get_test_data_pca(variance_ratios,                                                                                            evecs_all,                                                                                            f_tf_idf_vecs,                                                                                            f2_sim_mat,                                                                                            mapping,                                                                                            train_dict,                                                                                            test_dict, comb)


# In[48]:


get_table_per_feature_comb(threshold_var_list, f1_dim_list, f2_dim, alpha_list, train_rmse_list, test_rmse_list)


# In[49]:


index = test_rmse_list.index(min(test_rmse_list))
res1 = [threshold_var_list[index], f1_dim_list[index], alpha_list[index], test_rmse_list[index]]

with open("Results\\res_"+str(comb), 'wb') as bpt:
        pickle.dump(res1, bpt)


# In[53]:


get_plot(threshold_var_list, test_rmse_list, axis_names=['Variance ratio','RMSE'], title = 'Test RMSE vs Variance ratio of eigenvectors of F1 tf-idf vectors', dim_list = f1_dim_list)


# In[ ]:




