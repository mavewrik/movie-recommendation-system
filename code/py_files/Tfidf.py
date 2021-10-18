#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import csv
import ast
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random


# In[ ]:


path = 'the-movies-dataset'
filename = 'ratings.csv'
file = []
with open(path + '/' + filename, 'r', errors = 'ignore') as csvfile:
    spamreader = csv.DictReader(csvfile)
    for row in spamreader:
        file.append(row)


# In[3]:


len(file)


# In[4]:


file[0].keys()


# In[5]:


small_dataset_of_rating = []
for i in range(len(file)):
    temp = []
    for j in file[i].keys():
        temp.append(file[i][j])
    small_dataset_of_rating.append(temp)


# In[6]:


small_dataset_of_rating[0]


# In[12]:


small_rating_dict = {}
for i in range(len(small_dataset_of_rating)):
    movie = int(small_dataset_of_rating[i][1])
    user = int(small_dataset_of_rating[i][0])
    rating = float(small_dataset_of_rating[i][2])
    if user not in small_rating_dict.keys():
        small_rating_dict[user] = {movie: rating}
    else:
        small_rating_dict[user][movie] = rating
    


# In[13]:


small_rating_dict


# In[18]:


movies = []
for i in range(len(small_dataset_of_rating)):
    movies.append(small_dataset_of_rating[i][1])


# In[25]:


klm = list(set(movies))
klm.sort()


# In[26]:


klm


# In[46]:


import pickle
with open('train_and_test_dict_1', 'wb') as fp:
    pickle.dump(getit, fp)


# In[45]:


getit = [train ,test]


# In[36]:


len(train.keys())


# In[ ]:


import random


# In[ ]:





# In[40]:


def training(train_user_dict, sim1, sim2, mapping, start, incr): 
    a = []
    rmse = []
    for i in np.arange(start, 1.0, incr):
        error = []
        for j in train_user_dict.keys():
            error += user_pred_errors(train_user_dict[j], sim1, sim2, i, mapping)
        
        error = np.array(error)
        mean_all = np.mean(error)
        sv = mean_all ** 0.5
        rmse.append(sv)
        a.append(i)
        
    return a, rmse    


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


import pickle
rating_dict = pickle.load(open('rating_small_dict','rb'))


# In[43]:


len(rating_dict.keys())


# In[3]:


movie_dict = pickle.load(open('movie_ids','rb'))
movie_dict = list(movie_dict)


# In[4]:


len(movie_dict)


# In[8]:


len(rating_dict.keys())


# In[ ]:





# In[9]:


small_dict = {}
for i in rating_dict.keys():
    if len(small_dict.keys()) >= 100000:
        break
    temp = set(rating_dict[i].keys()).intersection(set(movie_dict))
    if len(temp) >= 10:
        for j in list(temp):
            if i not in small_dict.keys():
                small_dict[i] = { j : rating_dict[i][j] }
            else:
                small_dict[i][j] = rating_dict[i][j]
    


# In[13]:


small_dict


# In[ ]:





# In[11]:


my_data = pickle.load(open('movies_with_details_dic2','rb'))


# In[12]:


def give_doc(docs, g1, g2, a):
    l1 = []
    l2 = []
    mappings = []
    for i in docs.keys():
        mappings.append(i)
        movie = docs[i]
        temp_doc = ''
        for j in range(len(g1)):
            if g1[j] == 'actors':
                for k in range(a):
                    try:
                        if len(temp_doc) == 0:
                            temp_doc += movie[g1[j]][k]
                        else:
                            temp_doc += ' ' + movie[g1[j]][k]
                    except:
                        break
            elif g1[j] == 'overview':
                if len(temp_doc) == 0:
                    temp_doc += movie[g1[j]]
                else:
                    temp_doc += ' ' + movie[g1[j]]
            else:
                #print(movie[g1[j]])
                for lm in range(len(movie[g1[j]])):
                    if len(temp_doc) == 0:
                        temp_doc += movie[g1[j]][lm]
                    else:
                        temp_doc += ' ' + movie[g1[j]][lm]
        #print(temp_doc)
        l1.append(temp_doc)
        
        temp_doc = ''
        for j in range(len(g2)):
            if g2[j] == 'overview':
                if len(temp_doc) == 0:
                    temp_doc += movie[g2[j]]
                else:
                    temp_doc += ' ' + movie[g2[j]]
            else:
                for lm in range(len(movie[g2[j]])):
                    if len(temp_doc) == 0:
                        temp_doc += movie[g2[j]][lm]
                    else:
                        temp_doc += ' ' + movie[g2[j]][lm]
        
        l2.append(temp_doc)

    return l1,l2, mappings


# In[13]:


g1 = ['actors','directors','keyterms','genres']
g2 = ['overview']
a = 10


# In[ ]:





# In[14]:


l1, l2, mapping = give_doc(my_data, g1, g2, a)


# In[15]:


l1[0]


# In[16]:


l2[0]


# In[90]:


my_data[2]


# In[17]:


len(my_data.keys())


# In[ ]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(doc1)


# In[ ]:


Y = X.toarray()
Y.shape


# In[ ]:


transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(Y)


# In[ ]:


My_Vec = tfidf.toarray()
My_Vec.shape


# In[ ]:





# In[41]:


def split(user_dict, p):
    my_list = list(user_dict.keys())
    random.shuffle(my_list)
    upto = int(len(my_list)*p/100)
    tst = my_list[:upto]
    tr = my_list[upto:]
    train_set = {}
    test_set = {}
    for i in user_dict.keys():
        if i in tr:
            train_set[i] = user_dict[i]
        elif i in tst:
            test_set[i] = user_dict[i]
    
    return train_set, test_set


# In[44]:


train, test = split(rating_dict, 15)


# In[23]:


len(small_dict.keys())


# In[32]:


lst = list([1,2,3])

random.shuffle(lst)
print(lst)


# In[ ]:


from gensim.models.doc2vec import Doc2Vec


# In[ ]:


learning rate = 0.025
minimum_learning_rate = 0.025 
decrease_amount = 0.002
epochs = 10


# In[ ]:


model = Doc2Vec(alpha= learning rate, min_alpha= minimum_learning_rate)
model.build_vocab(sentences)
for epoch in range(epochs):
    model.train(sentences)
    model.alpha -= decrease_amount  # decrease the learning rate
    model.min_alpha = model.alpha


# In[ ]:





# In[ ]:


vec1 = model.infer_vector(fisrt_text.split())
vec2 = model.infer_vector(second_text.split())

similairty = spatial.distance.cosine(vec1, vec2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Now on 

# In[2]:


path = 'the-movies-dataset'
filename = 'keywords.csv'
file = []
with open(path + '/' + filename, 'r', errors = 'ignore') as csvfile:
    spamreader = csv.DictReader(csvfile)
    for row in spamreader:
        file.append(row)


# In[3]:


keywords_file = []
for i in range(len(file)):
    keywords_file.append([int(file[i]['id']), file[i]['keywords']])


# In[4]:




with open('Keyword', 'wb') as fp:
    pickle.dump(keywords_file, fp)


# In[107]:


def get_id_and_key(text):
    ret_data = []
    for i in range(len(text)):
        key_word_list = ast.literal_eval(text[i][1])
        id_n = int(text[i][0])
        ret_data.append([id_n, key_word_list])
    return ret_data


# In[108]:


keywords = get_id_and_key(keywords_file)


# In[109]:


len(keywords)


# In[110]:


keywords[0]


# In[5]:


path = 'the-movies-dataset'
filename = 'movies_metadata.csv'
file = []
with open(path + '/' + filename, 'r', errors = 'ignore') as csvfile:
    spamreader = csv.DictReader(csvfile)
    for row in spamreader:
        file.append(row)


# In[6]:


len(file)


# In[7]:


movies_overview = []
gone = []
for i in range(len(file)):
    try:
        id_m = int(file[i]['id'])
        overview = file[i]['overview']
        movies_overview.append([id_m, overview])
    except:
        gone.append(i)
        pass


# In[8]:


len(movies_overview)


# In[9]:


with open('metadata', 'wb') as fp:
    pickle.dump(file, fp)


# In[124]:


gone


# In[221]:


def get_only_keywords_for_w2v(kw):
    docs = []
    stop_words = list(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[a-zA-Z0-9]*[*.*]?\w+')
    for i in range(len(kw)):
        if len(kw[i][1]) > 0:
            temp = []
            for j in range(len(kw[i][1])):
                tokens = tokenizer.tokenize(kw[i][1][j]['name'])
                for k in tokens:
                    if k not in stop_words:
                        temp.append(k)
            if len(temp) > 0:
                docs.append(temp)
    return docs


# In[222]:


only_keywords_doc = get_only_keywords_for_w2v(keywords)


# In[228]:


def get_only_overview_for_w2v(text):
    docs = []
    stop_words = list(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[a-zA-Z0-9]*[*.*]?\w+')
    for i in range(len(text)):
        tokens = tokenizer.tokenize(text[i][1])
        temp_tk = []
        for j in tokens:
            if j not in stop_words:
                temp_tk.append(j)
        docs.append(temp_tk)
    return docs


# In[229]:


only_overview = get_only_overview_for_w2v(movies_overview)


# In[ ]:




