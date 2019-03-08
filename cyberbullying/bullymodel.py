import numpy as np
import nltk 
import pandas as pd 
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score

data = pd.read_csv(".\\bully.csv")
data = pd.DataFrame(data)
data = data.values

x = data[:,:-1]
y= data[:,len(data[0])-1]
y=y.astype('int')


 
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7,stop_words=stopwords.words('english'))  
x= vectorizer.fit_transform(x.ravel()).toarray()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 


classifier = RandomForestClassifier(n_estimators=50, random_state=0)  
classifier.fit(X_train, y_train)  

def predict_class(data)
    
    custom = CountVectorizer(vocabulary= vectorizer.vocabulary_)
    newdata = custom.fit_transform(data).toarray()
    y_pred = classifier.predict(newdata)
    return y_pred


