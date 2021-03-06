# Movie Recommender Using Deep Neural Network

This project applied keras embedding method and nueral networks to build a movies/TV recommender system with the data from Amazon reviews in 2018.

I also built an app to demo the model developed in this project. 

<A href='https://movie-recommender-max.herokuapp.com/'>Max-Your-Movie-Recommender</a>

note: Due to the compacity of github, the data in the demo only includes 5000 users/reviewers. Therefore, the movies recommended are only the ones that included in this smaller sub dataset. 


If you have any questions, please feel free to contact me:

daihongchen@icould.com

<a href="https://www.linkedin.com/in/daihong-chen-872538194/">Linkedin</a>


## Executive Summary

The goal of the project is to build a high performance movies/TV recommender system to improve personalized marketing, to engage Amazon customers, so as to drive the movies/TV products sales at Amazon.

Different from other products, online entertainment products such as movies and TV, are watched/purchased  by users mostly from recommendations. A high performance recommendation system that could identify the latent features of the movies and the users, is able to better personalize the recommended movies that are most likely interesting to each user. It is a critical way to enhance personalized marketing, to engage users, so as to drive sales.

Movie recommender using nueral networks has a few advantages compared to conventional approaches:

1. It is well-established that neural networks are able to deal with complex interaction patterns and precisely reflect the user’s preference.
2. Deep neural networks can efficiently learn the underlying explanatory factors and useful representations from input data.
3. Neural Collaborative Filtering is a representative work that constructs a dual neural network to model this two-way interaction between users and items.

By utilizing neural networks, the recommendation system would reach a high performance. In this study, the metrics used to evaluate the model performance is Mean Absolute Error. The final model was able to reach a Mean Absolute Error as low as 0.43. The corss validation on the training dataset validated the result with an average MAE of 0.39, and standard deviation of 0.012.

Though model performance is relatively high from the perspective of academics, a more practical evaluation is to conduct a AB test on the Amazon website and to investigate if the sales/revenues with the new recommender system would be significantly higher than the existing system.

## Navigation of the Repository

1. Folder - notebook, which contains four notebooks and the saved final model.

    a. final_report.ipynb is the final report notebook using source code for audiences to read.
    
    b. mr_model.h5 is the saved final model that can be loaded to make predictions.
    
    c. 01_create_subsample_2018.ipynb is the notebook that unzips, cleans, and subsamples the data.All code in this notebook were refactorized into get_data.py source file.
    
    d. 02_exploratory.ipynb is the notebook for exploration and visualization.All code in this notebook were refactorized into explore.py source file.
    
    e. 03_cf_model.ipynb shows the process of trying different models. The code for base model and final model were refactorized into cl_models.py source file.

2. Folder - src, which contains source code for the final_report.ipynb

    a. get_data.py includes funcitons to get the data for exploration, visualization, and modeling.
    
    b. explore.py includes functions to explore and visualize the data after get the data.
    
    c. cf_models.py includes functions to initiate base model and final model.
    
    d. cross_val.py includes funcitons to conduct cross validation on the trainning dataset.
    
    e. recommend_movies.py includes functions to make predictions/recommendations using saved final model.

3. Folder - images, which contains images used in the README.md file.
    
4. README.md, which is this file describing the project in details.

5. environment.yml, which can be used to create the environment (
<a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file">More help</a>): 

    
        conda env create -f environment.yml
    
6. Movie_Recommender_ppt.pdf, which is a pdf versin of the presentation of this project. It also can be used as memo.    


  
## Business Understanding

1. Online entertainment market is growing rapidly but also competitive. Personalized marketing is critical to drive sales of online entertainment products such as movies and TV.

2. Recommendation system plays a key role in marketing and sales in online commerce. 35% of Amazon.com’s revenue is generated by its recommendation engine through either onsite recommendation systems or offsite recommendation systems such as emails.(<a href="https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers">source</a>).

2. More than 80% of the TV shows people watch on Netflix are discovered through its onsite recommendation system (<a href='https://www.wired.co.uk/article/how-do-netflixs-algorithms-work-machine-learning-helps-to-predict-what-viewers-will-like'>source, 2017</a>).

3. An effective recommender system drives highly personalized marketing for the Amazon movies/tv products, so as to better engage customers and increase sales. 

4. The machine learning technique and algorithm used in this recommender could be transferred to other Amazon products. 


## Data Understanding

#### Download the data
The data used in this project can be downloaded <a href="https://nijianmo.github.io/amazon/index.html">HERE</a>. The Movies and TV reviews are one of the amazon products and you can click the reviews and metadata to download the data. The website requires the user to fill a form before the download. Once you submit the form, the data will be downloaded immediately. 

<a href="http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz">Movies and TV reviews</a> (8,765,568 reviews) & <a href="http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz">Metadata</a> (203,970 products)

You can also use !wget to download the data:

!wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz

!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz

The downloaded data files are Json.gz file. 

Create a new folder named data. The downloaded data will be store in this folder.


#### Data preparation

Data preparation is used the source code in src.get_data.py.

1. Json.gz file with 19 years data (8,765,568 reviews) 
2. Unzip json.gz file and transform into pandas dataframe.
2. Extracted a subsample that only contains 2018 ratings/reviews (209,060 reviews) because of computational cost. Susample is saved as a csv file named data_2018_mr.csv in the data folder. so that I do not have to run the time-consuming process again.
3. Scrape review webpage link for each movie from the variable-"details" using BeautifulSoup.
4. Drop unrelated variables. Building a collaborative filtering model only requires three features of ratings, reviewerIDs, and movieIDs. Some related variables for deployment are also selected. The rest are dropped.
5. The variables in the preprocessed dataset are: 

        'rating', 'reviewTime', 'reviewerID', 'movieID', 'style',  'reviewerName',
        'reviewText', 'summary', 'unixReviewTime', 'vote', 'category', 'title',
        'main_cat', 'description', 'brand', 'price',  'links', 'reviewer_count',
        'movie_count', 'average_rating'. 

#### Data exploration and findings
Explore and visualize the data using the source code in src.explore.py.

1. Majority ratings are 5.
2. 50% reviewers only reviewed 1 movie.
3. 25% movies have only one review.
4. Data is sparse. The initial exploration found that more than 50% of reviewers only reviewed 1 item, and the model performance is very low. Therefore, reviews from reviewers that only reviewed 1 movie and movies that only got one review were removed. 


<img src="images/4.png" width='800' /> ![](images/k8s-dashboard.png)


<img src="images/2.png" width=800 /> ![](images/k8s-dashboard.png)

<img src="images/3.png" width=6000 /> ![](images/k8s-dashboard.png)


## Model Understanding

### Neural Collaborative Filtering Model 

#### What and Why?

The movie recommender was a Collaborative Filtering model with deep learning embedding technique. Collaborative filtering model uses similarities of users' preferences to predict the movies/TV that a given user has not watched/purchased, but are highly interesting to this user. It is a model based recommender.

The model applies deep learning keras embedding technique. Embedding is split one matrix into two smaller matrix, or transform high dimension to low dimensions. Embedding is one notable successful use of deep learning to represent discrete variables as continuous vectors. The movies embeddings creates a low dimensional space in which the movies that have been watched by a given user are nearby in the 'movie preference' space, and the users embeddings are closer to the movies that they have watched. These individual dimensions in these vectors typically have no inherent meaning. Instead, it’s the overall patterns of location and distance between vectors that machine learning takes advantage of. So that the model is able to recommend other movies based on those movies' proximity to a user embedding, because nearby users and movies share preferences.

 
#### Base model:

1. Create the reviewer embeddings and movie embeddings as input layer.When create an Embedding layer, the weights for the embedding are randomly initialized (just like any other layer).

2. Uae Dot.product to merge two embeddings on reviewers and movies as output layer. 
    
Base model did not perform well with the loss function of Squared Mean Error as 12, and the metrics, Mean Absolute Error as 2.4.

#### Final Model:

1. Create the reviewer embedding and movie embedding as input layers.When create an Embedding layer, the weights for the embedding are randomly initialized (just like any other layer), and are gradually adjusted via backpropagation during training.
    
2. Use concatenate to merge embedding layers: It takes as input a list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor, the concatenation of all inputs (https://keras.io/layers/merge/).

3. Add hidden layers. Hidden layers that better learn the underlying factors and representations to adjust the weights via backpropagation.

4. Add dropout to help with preventing overfitting on training dataset.

Final model performs very well with the loss function of Squared Mean Error as 0.78, and the metrics, Mean Absolute Error as 0.43.


## Model Evaluation: 

Metrics: Mean Absolute Error (MAE)

Mean Absolute Error (MAE) measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight. Due to the predicted value is the ratings ranging from 1 to 5, and being as 5 could be consider as good as 5 times of being as 1, and because there is no need to penalize the outliers, MAE is more appropriated, and easier to interpret. 

For example, if the model predicts 3 to a movie for a given user, this means that on average the true rating is between 2.6 and 3.4

The base model only includes input and output layers. Hidden layers and dropout(help with preventing overfitting) were added in the final model.

    The MAE of the base model is 2.4.

    The MAE of final model is 0.43.

Model performance was considerably good compared MAE in conventional models which is aorund 0.7(<a href="https://doi.org/10.1016/j.eij.2016.10.002">citation</a>). It would help improve Amazon's personalized marketing on movies/TV products.

### Cross validation:
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


sources:

https://www.tensorflow.org/tutorials/text/word_embeddings

https://keras.io/layers/merge/

https://medium.com/spikelab/learning-embeddings-for-your-machine-learning-model-a6cb4bc6542e

https://medium.com/@jdwittenauer/deep-learning-with-keras-recommender-systems-e7b99cb29929

https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers

https://doi.org/10.1016/j.eij.2016.10.002
