import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import pickle
import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D,  Dropout, Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools

# import data
def explore_data():
    data_file = '../data/2018_movie_links.csv'
    data_2018 = pd.read_csv(data_file)
    data_2018 = data_2018.drop(['Unnamed: 0', 'verified', 'rank', 'also_buy', 'also_view', 'details'], axis=1)
    data_2018_ratings = data_2018[['overall', 'asin', 'reviewerID', 'title']]
    ratings_2018 = data_2018_ratings.rename(columns={'overall':'rating', 'asin':'productID'})
    reviewer_count = ratings_2018.groupby('reviewerID')['rating'].count()
    product_count = ratings_2018.groupby('title')['rating'].count()
    top_reviewers = reviewer_count.sort_values(ascending=False)[:20]
    count_reviewers = len(reviewer_count)
    top_products = product_count.sort_values(ascending=False)[:20]

    print(f"Count of Reviewers: {len(reviewer_count)}")
    print(f"Count of Products: {len(product_count)}")
    print("")
    print(f"Ratings descriptive statistics: ")
    print(ratings_2018['rating'].describe()) 
    print('')
    print(f"Reviewers by count descriptive statistics: ")
    print(reviewer_count.describe())
    print("")
    print(f"Products by count descriptive statistics: ")
    print(product_count.describe())
    print("")
    print(f"Top reviwers by count of reviews: ")
    print(top_reviewers)
    print("")
    print(f"Top movies by count of reviews: ")
    print(top_products) 

def sub_dataset():
    data_file = '../data/2018_movie_links.csv'
    data_2018 = pd.read_csv(data_file)
    data_2018 = data_2018.drop(['Unnamed: 0', 'verified', 'rank', 'also_buy', 'also_view', 'details'], axis=1)
    data_2018_ratings = data_2018[['overall', 'asin', 'reviewerID', 'title']]
    ratings_2018 = data_2018_ratings.rename(columns={'overall':'rating', 'asin':'productID'})
    reviewer_count = ratings_2018.groupby('reviewerID')['rating'].count()
    product_count = ratings_2018.groupby('title')['rating'].count()
    top_reviewers = reviewer_count.sort_values(ascending=False)[:20]
    count_reviewers = len(reviewer_count)
    top_products = product_count.sort_values(ascending=False)[:20]
    
    return ratings_2018, top_reviewers, top_products


def create_dis_rating(ratings_2018):
    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(ratings_2018['rating'])
    plt.title("Distribution of Ratings")
    plt.xlabel('Ratings')
    plt.ylabel('Counts')
    plt.show()
    return plt.show()

def top_reviewers(top_reviewers):
    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    top_reviewers.sort_values().plot(kind='barh',color='#86bf91', zorder=2, width=0.85)
    plt.title('Top Reviewers')
    
    plt.show()

def top_products(top_products):
    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

    top_products.sort_values().plot(kind='barh', zorder=2, color='purple', width=0.85)
    plt.title('Top Movies')

    plt.show()
