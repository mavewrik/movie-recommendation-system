#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


needed = defaultdict(dict)

with open('./data/ratings.csv', 'r') as f:
    next(f)
    lines = (line.rstrip() for line in f)
    lines = list(line for line in lines if line)
    for line in lines:
        splits = list(map(float, line.split(',')))
        needed[int(splits[0])].update({int(splits[1]):splits[2]})


# In[23]:


joblib.dump(needed, 'nested_dict.sav')


# In[ ]:





# In[ ]:





# In[2]:


overview_docs_tf_idf = joblib.load('./overview_docs_tfidf1')
actor_dir_genr_key_docs_tf_idf = joblib.load('./actor_dir_genr_key_docs_tfidf1')


# In[3]:


def plot_tsne(X):

    tsne = TSNE(n_components=2).fit_transform(X)

    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()


# In[5]:


plot_tsne(overview_docs_tf_idf)


# In[4]:


plot_tsne(actor_dir_genr_key_docs_tf_idf)


# In[ ]:




