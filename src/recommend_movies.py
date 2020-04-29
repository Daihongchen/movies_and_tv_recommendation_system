from tensorflow.keras.models import load_model
import pandas as pd 
import numpy as np

def recommender(test_value):
    ## pull the data for deploy
    file = '../data/data_2018_mr.csv'
    data_mr = pd.read_csv(file)
    model_path = '../notebook/new_final_model.h5'
    model_mr = load_model(model_path)
    movie_data = np.array(list(set(data_mr.movie)))
    user = np.array([test_value for i in range(len(movie_data))])
    
    predictions = model_mr.predict([user, movie_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_movie_ids = (-predictions).argsort()[:5]

    recommend = data_mr[data_mr['movie'].isin(recommended_movie_ids)]
    recommend = recommend[['movie',
                            'title', 
                            'average_rating',
                            'category', 
                            'description', 
                            'price', 
                            'links']].drop_duplicates()
    
    return recommend