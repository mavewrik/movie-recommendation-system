#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import ast
import math
import pickle
import numpy as np 
import contractions
from nltk.corpus import stopwords
from sklearn.externals import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer('[A-Za-z0-9]?\w+')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[2]:


meta_data = pickle.load(open("Dataset\\metadata",'rb'))
Keyword = pickle.load(open("Dataset\\Keyword",'rb'))


# In[3]:


user_ratings = pickle.load(open("rating_small_dict1",'rb'))
movie_ids = []
for user in user_ratings.keys():
    movie_ids += list(user_ratings[user].keys())


# In[4]:


movie_ids = set(movie_ids)


# In[5]:


len(movie_ids)


# In[6]:


movies = dict()
for movie_id in movie_ids:
    movies[movie_id] = dict()


# In[7]:


def normalize(doc):
    doc = contractions.fix(doc)
    tokens = tokenizer.tokenize(doc)
    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        if token == '' or token == '.' or token == '_' or token in stop_words:
            continue
        token = lemmatizer.lemmatize(token)
        normalized_tokens.append(token)
    return normalized_tokens


# In[8]:


def get_movies_with_keytems(keyword_doc, movies):
    for row in keyword_doc:
        movie_id = int(row[0])
        if movies.get(movie_id) is None:
            continue
        keyterms = list()
        movie_keyterm_dic_list = ast.literal_eval(row[1])
        for movie_keyterm_dic in movie_keyterm_dic_list:
            keyterms.append(movie_keyterm_dic['name'])
        movies[movie_id]["keyterms"] = keyterms
    return movies


# In[9]:


movies_with_keyterms = get_movies_with_keytems(Keyword, movies)


# In[10]:


# c = 0
# for k in movies_with_keyterms.keys():
# #     print(movies_with_keyterms[k])
#     if movies_with_keyterms[k].get("keyterms") is None:
#         c += 1
# c


# In[11]:


len(movies_with_keyterms)


# In[12]:


def get_movie_with_actors_directors(movies):
    with open("Dataset\\credits.csv", 'r', errors = 'ignore') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            movie_id = int(row['id'])
            if movies.get(movie_id) is None:
                continue
            actors = list()
            cast = ast.literal_eval(row["cast"])
            for actor in cast:
                actors.append(actor['name'].replace(" ", "").lower())
            movies[movie_id]["actors"] = actors    
            crew = ast.literal_eval(row["crew"])
            directors = list()
            for member in crew:
                if member['job'].lower() == "director":
                    directors.insert(0, member['name'].replace(" ", "").lower())
                elif 'director' in member['job'].lower():
                    name = member['name'].replace(" ", "").lower()
                    directors.append(name)
            movies[movie_id]["directors"] = directors    
    return movies


# In[13]:


movies_with_keyterms_actors_directors = get_movie_with_actors_directors(movies_with_keyterms)


# In[14]:


# c = 0
# ids = []
# for k in movies_with_keyterms_actors_directors.keys():
# #     print(movies_with_keyterms_actors_directors[k])
#     if movies_with_keyterms_actors_directors[k].get("actors") is None:
#         ids.append(k)
#         c += 1
# c


# In[15]:


len(movies_with_keyterms_actors_directors)


# In[16]:


def get_movie_genre_descr(movies, meta_data):
    for row in meta_data:
        if not row['id'].isdecimal():
            continue
        else:
            movie_id = int(row['id'])
        if movies.get(movie_id) is None:
            continue
        if row.get("overview") is not None:
            movies[movie_id]["overview"] = row.get("overview")
        else:
            movies[movie_id]["overview"] = []
        genres = list()
        gs = ast.literal_eval(meta_data[0]["genres"])
        for genre in gs:
            genres.append(genre['name'].replace(" ", "").lower())
        movies[movie_id]["genres"] = genres
    return movies


# In[17]:


movies_with_details_dic1 = get_movie_genre_descr(movies_with_keyterms_actors_directors, meta_data)


# In[18]:


len(movies_with_details_dic1)


# In[20]:


with open("movies_with_details_dic1", 'wb') as fp:
    pickle.dump(movies_with_details_dic1, fp)


# In[26]:


# movies_with_details_dic1[5]


# In[21]:


all_doc_1 = pickle.load(open("all_doc_1",'rb'))


# In[22]:


actor_dir_genr_key_docs1, overview_docs1, mapping1 = all_doc_1[0], all_doc_1[1], all_doc_1[2] 


# In[23]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(actor_dir_genr_key_docs1)

Y = X.toarray()
print(Y.shape)

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(Y)

actor_dir_genr_key_docs_tfidf1 = tfidf.toarray()
print(actor_dir_genr_key_docs_tfidf1.shape)


# In[24]:


np.unique(actor_dir_genr_key_docs_tfidf1[0])


# In[25]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(overview_docs1)

Y = X.toarray()
print(Y.shape)

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(Y)

overview_docs_tfidf1 = tfidf.toarray()
print(overview_docs_tfidf1.shape)


# In[26]:


np.unique(overview_docs_tfidf1[0])


# In[27]:


with open("actor_dir_genr_key_docs_tfidf1", 'wb') as fp:
    pickle.dump(actor_dir_genr_key_docs_tfidf1, fp)


# In[28]:


with open("overview_docs_tfidf1", 'wb') as fp:
    pickle.dump(overview_docs_tfidf1, fp)


# In[29]:


actor_dir_genr_sim_matr1 = cosine_similarity(actor_dir_genr_key_docs_tfidf1, actor_dir_genr_key_docs_tfidf1)


# In[30]:


actor_dir_genr_sim_matr1.shape


# In[31]:


overview_sim_matr1 = cosine_similarity(overview_docs_tfidf1, overview_docs_tfidf1)


# In[36]:


overview_sim_matr1[0]


# In[33]:


overview_sim_matr1.shape


# In[37]:


with open("actor_dir_genr_sim_matr1", 'wb') as fp:
    pickle.dump(actor_dir_genr_sim_matr1, fp)


# In[38]:


with open("overview_sim_matr1", 'wb') as fp:
    pickle.dump(overview_sim_matr1, fp)


# In[ ]:




