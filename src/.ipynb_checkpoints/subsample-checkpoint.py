import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

def create_subsample():
    !wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz
    !wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz
    data_meta = []
    with gzip.open('meta_Movies_and_TV.json.gz') as f:
        for l in f:
        data_meta.append(json.loads(l.strip()))
    
    df_meta = pd.DataFrame.from_dict(data_meta)

# drop useless columns
    df_meta_1 = df_meta.dropna(subset=['details'])
    df_meta_1 = df_meta_1.drop(['image', 'feature', 'date', 'tech1'], axis=1)
    data_movie = []
    with gzip.open('Movies_and_TV.json.gz') as f:
        for l in f:
            data_movie.append(json.loads(l.strip()))
    
    df_movie = pd.DataFrame.from_dict(data_movie)

    df_movie_1 = df_movie[df_movie['reviewTime'].str.contains('2016') 
    |df_movie['reviewTime'].str.contains('2017') 
    |df_movie['reviewTime'].str.contains('2018')]

    df_movie_1 = df_movie_1.drop('image', axis=1)
    df_movie_1 = df_movie_1[df_movie_1['verified']==True]
    df_movie_meta = df_movie_1.merge(df_meta_1, on='asin', how='inner')
    df_movie_2018 = df_movie_meta[df_movie_meta['reviewTime'].str.contains('2018')]

    links = []
    for i in movie_2018['details']:
        soup = BeautifulSoup(i)
        found_links = soup.select('a.a-text-normal')
        if found_links:
            link = found_links[0]['href']
            links.append(link)
        else:
            links.append("")
    movie_2018['links'] = links
    # movie_2018.to_csv('2018_movie_links.csv')
    return movie_2018


def create_dataset_model():
    data_2018 = pd.read_csv(data_file)
    data_2018 = data_2018.drop(['Unnamed: 0', 'verified', 'rank', 'also_buy', 'also_view', 'details'], axis=1)
    data_2018 = data_2018.rename(columns={'overall':'rating', 'asin':'movieID'})
    reviewer_count = data_2018.groupby('reviewerID')['rating'].count()
    product_count = data_2018.groupby('movieID')['rating'].count()
    average_rating = data_2018.groupby('movieID')['rating'].mean()
    # remove reviewers that has only one review.
    data_2018_1 = data_2018.merge(reviewer_count, on='reviewerID')
    data_2018_1 = data_2018_1.rename(columns={'rating_y':'reviewer_count', 'rating_x':'rating'})
    data_2018_1 = data_2018_1.merge(product_count, on='movieID')
    data_2018_1 = data_2018_1.rename(columns={'rating_y':'movie_count', 'rating_x':'rating'})
    data_2018_1 = data_2018_1.merge(average_rating, on='movieID')
    data_2018_1 = data_2018_1.rename(columns={'rating_y':'average_rating', 'rating_x':'rating'})
    data_2018_1 = data_2018_1[data_2018_1['reviewer_count']>1]
    data_2018_1 = data_2018_1[data_2018_1['movie_count']>1]

    reviewer_enc = LabelEncoder()
    data_2018_1['reviewer'] = reviewer_enc.fit_transform(data_2018_1['reviewerID'].astype(str).values)
    movie_enc = LabelEncoder()
    data_2018_1['movie'] = movie_enc.fit_transform(data_2018_1['movieID'].astype(str).values)
    data_2018_1['rating'] = data_2018_1['rating'].values.astype(np.float32)
    
    data_2018_1.to_csv('data_2018_mr.csv')
