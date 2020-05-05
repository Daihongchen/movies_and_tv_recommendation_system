# Movie Recommender Using Deep Neural Networks

In this project, I applied keras embedding method and nueral networks to build a movies/TV recommender system with the data from Amazon reviews in 2018.

If you have any questions, please feel free to contact me:

daihongchen@icould.com

<a href="https://www.linkedin.com/in/daihong-chen-872538194/">Linkedin</a>


## Executive Summary

The project goal is to build a high performance movies/TV recommender system to engage Amazon customers, so as to drive the movies/TV products sales.

Different from other products, online entertainment products such as movies and TV, are watched by users mostly from recommendations. A high performance recommendation system that could identify the latent features of the movies and the users, so as to better personalize the recommendaded movies that are most likely interesting to each user. It is very important to attract and egage users, so as to drive sales.

Movie recommender using nueral networks has a few advantages compared to conventional approaches:

1. It is well-established that neural networks are able to deal with complex interaction patterns and precisely reflect the user’s preference.

2. Deep neural networks can efficiently learn the underlying explanatory factors and useful representations from input data.

3. Neural Collaborative Filtering is a representative work that constructs a dual neural network to model this two-way interaction between users and items.

By utilizing neural networks, the recommendation system would reach a high performance. In this study, the metrics used to evaluate the model performance is Mean Absolute Error. The final model reached 0.43 of the MAE. 


## Data

The data used in this project can be downloaded <a href="https://nijianmo.github.io/amazon/index.html">HERE</a>. The Movies and TV reviews are one of the amazon products and you can click the reviews and metadata to download the data. The website requires the user to fill a form before the download. Once you submit the form, the data will be downloaded immediately. 

Movies and TV

reviews (8,765,568 reviews)

metadata (203,970 products)

Or you can use the following codes for downloading the data:

!wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz

!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz

The downloaded data files are Json.gz file. The files are unzipped  and processed using the codes in the get_data.py file. The following are some keypoints about the data.


    1. Json.gz file with 19 years data (8,765,568 reviews) 
    2. Subsample to 2018 ratings/reviews (209,060 reviews)
    3. Scrape review webpage link for each movie from the data
    4. Drop unrelated variables
    5. Preprocessed data features include: 
    'rating', 'reviewTime', 'reviewerID', 'movieID', 'style',  'reviewerName',
    'reviewText', 'summary', 'unixReviewTime', 'vote', 'category', 'title', 'main_cat',
    'description', 'brand', 'price',  'links', 'reviewer_count', 'movie_count', 
    'average_rating'
        
## Data Preparation and Exploration

1. Subsampled the data to only include 2018 reviews.
2. Clean the data. 
3. Explore the data

### Findings

1. Majority ratings are 5.
2. 50% reviewers only reviewed 1 movie.
3. 25% movies have only one review.
4. Data is sparse. The initial exploration found that more than 50% of reviewers only reviewed 1 item, and the model performance is very low. Therefore, reviews from reviewers that only reviewed 1 movie and movies that only got one review were removed. 


<img src="images/4.png" width='800' /> ![](images/k8s-dashboard.png)


<img src="images/2.png" width=800 /> ![](images/k8s-dashboard.png)

<img src="images/3.png" width=6000 /> ![](images/k8s-dashboard.png)


## Neural Networks Model

Collaborative Filtering
 
Keras Embedding
    Create the reviewer embeddings and movie embeddings 
        Discomposing the utility matrix 
    Dot.product to merge two embeddings on reviewers and movies
    Add hidden layers

Metrics: MAE 

### Model results: 
The base model only includes input and output layers. Hidden layers and dropout(help with preventing overfitting) were added in the final model.

    The MAE of the base model is 2.4.

    The MAE of final model is 0.43.

Model performance was considerably good.

## Cross validation:
    Metrics: Mean Absolute Error (MAE)
    
    StratifiedKFold, n_splits=5 
    
        0.41, 
        0.41, 
        0.38, 
        0.37, 
        0.38
        
    Average 0f MAE: 0.39
    
    Standard Deviation of MAE:  0.012



### Final model Loss (loss function: Mean Squared Error)

<img src="images/1.png" width=800 /> ![](images/k8s-dashboard.png)


### Base model Loss (loss function: Mean Squared Error)



<img src="images/base_model.png" width=800 /> ![](images/k8s-dashboard.png)


### An exaple of the recommender


<img src="images/example1.png" width=800 /> ![](images/k8s-dashboard.png)


## Conclusion

Using neural networks in recommendation system improves the performance. This model could also transfer to other products. 

The majority of the ratings are 5 starts, so the data is skewed. In the next steps, I will conduct a sentiment analysis on the reviews and use the sentiment score instead of ratings to build the model. Or combine the ratings and sentiment score as the new ratings to build the model to compare the performance. 
