from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import numpy
import pandas as pd

def cross_val_v1(X_train, y_train):
    model_cv = load_model('../notebook/mr_model.h5')
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvscores = []
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_s, X_test_s = X_train[train_index], X_train[test_index]
        y_train_s, y_test_s = y_train[train_index], y_train[test_index]
        X_train_array = [X_train_s[:, 0], X_train_s[:, 1]]
        X_test_array = [X_test_s[:, 0], X_test_s[:, 1]]
#       # create model
        model_cv.fit(x=X_train_array, 
                    y=y_train_s, 
                    epochs=5, 
                    verbose=1,
                    validation_data=(X_test_array, y_test_s))
    #evaluate the model
        scores = model_cv.evaluate(x=X_test_array, y=y_test_s, verbose=1)
        print(model_cv.metrics_names[1], scores[1])
        cvscores.append(scores[1])
    cvscore_mean= numpy.mean(cvscores)
    cvscore_std = numpy.std(cvscores)

    return cvscores, cvscore_mean, cvscore_std

# def create_X_y():
    
#     file =  '../data/data_2018_mr.csv'
#     data_2018 = pd.read_csv(file)
    
#     X = data_2018[['reviewer','movie']].values
#     y = data_2018['rating'].values
    
#     return X, y

