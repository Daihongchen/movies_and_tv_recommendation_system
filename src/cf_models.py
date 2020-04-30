import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## final model
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras.layers import Add, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



# Final model
class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x

def RecommenderNet(n_reviewers, n_movies, n_factors, min_rating, max_rating):
    reviewer = Input(shape=(1,))
    r = EmbeddingLayer(n_reviewers, n_factors)(reviewer)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, n_factors)(movie)
    
    x = Concatenate()([r, m])
    x = Dropout(0.05)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[reviewer, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    return model

## base model
def RecommenderV4(n_reviewers, n_movies, n_factors):
    reviewer = Input(shape=(1,))
    r = Embedding(n_reviewers, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(reviewer)
    r = Reshape((n_factors,))(r)
    
    movie = Input(shape=(1,))
    m = Embedding(n_movies, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(movie)
    m = Reshape((n_factors,))(m)
    
    x = Dot(axes=1)([r, m])
    model = Model(inputs=[reviewer, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    return model


def train_test(data_2018):
    X = data_2018[['reviewer','movie']].values
    y = data_2018['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]
    
    return X_train_array, X_test_array, y_train, y_test


def create_parameters_model(data_2018):
    n_reviewers = data_2018['reviewer'].nunique()
    n_movies = data_2018['movie'].nunique()
    n_factors = 50
    min_rating = min(data_2018['rating'])
    max_rating = max(data_2018['rating'])

    
    return n_reviewers, n_movies, n_factors, min_rating, max_rating
    
    

def import_data():
    
    file =  '../data/data_2018_mr.csv'
    data_2018 = pd.read_csv(file)
    
    return data_2018



    