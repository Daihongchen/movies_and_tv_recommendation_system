# Movie Recommender Using Deep Neural Networks

In this project, I applied keras embedding method and nueral networks to build a movies/TV recommender system with the data from Amazon reviews in 2018.

The movie system could be found here:

http://0.0.0.0:5000/

If you have any questions, please feel free to contact me:

daihongchen@icould.com

<a href="https://www.linkedin.com/in/daihong-chen-872538194/">Linkedin</a>


## Executive Summary

Different from other products, movies or TV are watched by users mostly from recommendations. An high performance recommendation system that could identify the latent features of the movies and the users, so as to better personalize the recommendations to each user is very important to attract and egage users. 

Movie recommender using nueral networks has a few advantages compared to conventional approaches:

1. It is well-established that neural networks are able to approximate any continuous function with arbitrary precision by varying the activation choices and combinations. This property makes it possible to deal with complex interaction patterns and precisely reflect the user’s preference.

2. Deep neural networks can efficiently learn the underlying explanatory factors and useful representations from input data.

3. A recommendation can be viewed as a two-way interaction between users’ preferences and items’ features. For example, matrix factorization decomposes the rating matrix into low-dimensional user/item latent factors. Neural Collaborative Filtering is a representative work that constructs a dual neural network to model this two-way interaction between users and items.

By utilizing neural networks, the recommendation system would reach a high performance. In this study, the metrics used to evaluate the model performance is Mean Absolute Error. The final model reached a 0.49 of the MAE. 


## Data

The data used in this project can be downloaded <a href="https://nijianmo.github.io/amazon/index.html">HERE</a>

    1. Json.gz file with 19 years data (8,765,568 reviews) 
    2. Subsample to 2018 ratings/reviews (209,060 reviews)
    3. Scrape review webpage link for each movie from the data
    4. Drop unrelated variables
    5. Preprocessed data features include: 'rating', 'reviewTime', 'reviewerID', 'movieID', 'style',
       'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote',
       'category', 'title', 'main_cat', 'description', 'brand', 'price',
       'links', 'reviewer_count', 'movie_count', 'average_rating'
  
## Process

1. subsample
2. clean the data
3. explore the data

### Findings

1. Majority ratings are 5.
2. 50% reviewers only reviewed 1 movie.
3. 25% movies have only one review.
4. Data is sparse.

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

## Model results: 
MAE decreased from 2.4 to 0.5

## Cross validation:
    Metrics: Mean Absolute Error (MAE)
    
    StratifiedKFold, n_splits=5 
    MAE for each fold:
        0.398, 
        0.402, 
        0.399, 
        0.399, 
        0.399
        
    Average 0f MAE: 0.40
    
    Standard Deviation of MAE:  0.00131


### Final model Loss (loss function: Mean Squared Error)

<img src="images/1.png" width=800 /> ![](images/k8s-dashboard.png)


### Base model Loss (loss function: Mean Squared Error)



<img src="images/base_model.png" width=800 /> ![](images/k8s-dashboard.png)


### An exaple of the recommender


<img src="images/example1.png" width=800 /> ![](images/k8s-dashboard.png)


## Conclusion

Using neural networks in recommendation system improves the performance. This model could also transfer to other products. 
