import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_data():

    data_meta = []
    with gzip.open('../data/meta_Movies_and_TV.json.gz') as f:
        for l in f:
            data_meta.append(json.loads(l.strip()))
    
    df_meta = pd.DataFrame.from_dict(data_meta)

# drop useless columns
    df_meta = df_meta.dropna(subset=['details'])
    df_meta = df_meta.drop(['image', 'feature', 'date', 'tech1'], axis=1)
    
    # unzip movie data file and turn into dataframe
    data_movie = []
    with gzip.open('../data/Movies_and_TV.json.gz') as f:
        for l in f:
            data_movie.append(json.loads(l.strip()))
    
    df_movie = pd.DataFrame.from_dict(data_movie)
    # subsample to only include 2018 for computational cost consideration
    df_movie_2018 = df_movie[df_movie['reviewTime'].str.contains('2018')] 

    df_movie_2018 = df_movie_2018.drop('image', axis=1)
    df_movie_2018 = df_movie_2018[df_movie_2018['verified']==True]
    data_2018 = df_movie_2018.merge(df_meta, on='asin', how='inner')
    
    # extract links from column of 'details'
    links = []
    for i in data_2018['details']:
        soup = BeautifulSoup(i)
        found_links = soup.select('a.a-text-normal')
        if found_links:
            link = found_links[0]['href']
            links.append(link)
        else:
            links.append("")
    # add links back to dataframe
    data_2018['links'] = links

    # clean the data and proprecessing for exploration and model building
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

    # create transformed features for building models
    reviewer_enc = LabelEncoder()
    data_2018_1['reviewer'] = reviewer_enc.fit_transform(data_2018_1['reviewerID'].astype(str).values)
    movie_enc = LabelEncoder()
    data_2018_1['movie'] = movie_enc.fit_transform(data_2018_1['movieID'].astype(str).values)
    data_2018_1['rating'] = data_2018_1['rating'].values.astype(np.float32)
    
    return data_2018_1