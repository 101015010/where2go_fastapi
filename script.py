#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf




# # prédictions des notes utilisateurs avec des catégories

# In[2]:


"""import tensorflow_hub as hub"""


# In[3]:


"""model_lang = hub.load("universal-sentence-encoder_4.tar/universal-sentence-encoder_4")"""


# In[4]:
"""data = pd.read_csv('tourisme.csv')  """


# In[5]:


class model_notes :
    def __init__(self):  ## matrix doit contenir un champ id pour les utilisateurs
        self.cluster=pd.DataFrame()
        self.destinations=pd.DataFrame()
        self.kmeans=KMeans(n_clusters=0,random_state=0)
    """def fit_cluster(self):   ## cette méthode créer les clusters, à n'utiliser qu'une seule fois 
        data=self.destinations
        data["train"] = data['Nom_du_POI']+" "+data["Description"]
        data=data[~pd.isna(data["train"])]
        data_clus_trans = np.array(model_lang(data["train"]))
        y=self.kmeans.fit_predict(data_clus_trans)
        data["categorie"]=y
        data=data.sort_values("categorie").reset_index()
        self.cluster=data
    def predict_cluster(self,new_data):
        new_data["train"] = new_data['Nom_du_POI']+" "+new_data["Description"]
        new_data=new_data[~pd.isna(new_data["train"])]
        data_clus_trans = np.array(model_lang(new_data["train"]))
        self.kmeans.predict(data_clus_trans)
        new_data["categorie"]=self.kmeans.predict(data_clus_trans)
        new_data=new_data.sort_values("categorie").reset_index()
        return new_data"""
    def fit_predict_ratings(self, user_article_matrix, user_id):
        # Préparation des données
        mat = pd.DataFrame(user_article_matrix, columns=[str(k) for k in range(user_article_matrix.shape[1])])
        mat['id'] = mat.index
        mat_long = mat.melt(id_vars=['id'], var_name='categorie', value_name="note")
        mat_long = mat_long[mat_long['note'] != 0]

        # Chargement des données dans Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(mat_long[['id', 'categorie', 'note']], reader)
        trainset = data.build_full_trainset()
        # Entraînement de l'algorithme SVD
        algo = SVD()
        algo.fit(trainset)

        # Prédiction pour les notes manquantes
        predictions = []
        for item in range(user_article_matrix.shape[1]):
            prediction = algo.predict(str(user_id), str(item))
            predictions.append(prediction)

        return predictions



