# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:16:30 2019

@author: Yaya Liu
"""

import pandas as pd
import numpy as np
import random
import re
import nltk
import string

import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
#from nltk import PorterStemmer as Stemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
stopwords_forCloud = set(STOPWORDS)
#stopwords.update(['flight', 'flights', 'Flightled'])


import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(500) 

airline = pd.read_csv('Tweets.csv')
airline.shape              # (rows: 14640, columns:15)
#print(airline.head(5))


# subset the colums that will be used for analysis
airline_sub = airline.loc[:, ['airline_sentiment', 'airline', 'name', 'negativereason', 'text', 'tweet_created', 'tweet_location']]
airline_sub.isnull().sum()    # checking missing values of each column  
#print(airline_sub.shape)   # (rows: 14640, columns: 7)
    
sentiment_count = airline_sub['airline_sentiment'].value_counts()  # negative 9178, neutral 3099, positive 2363
print("airline_sub sentiment_count: ", sentiment_count)
    
airline_sub['label'] = airline_sub['airline_sentiment'].factorize()[0]  # add column "label". neutral 0, positive 1, negative 2
airline_sub['text_length'] = airline_sub['text'].apply(len)             # add a column to store the length of each tweet
#airline_sub.head(5)

def data_resample(airline_sub):
    pos = airline_sub.loc[airline_sub['airline_sentiment'] == "positive"]   # 2363 rows   
    len_pos = len(pos)
    
    neu = airline_sub.loc[airline_sub['airline_sentiment'] == "neutral"]  # 3099 rows   
    neg = airline_sub.loc[airline_sub['airline_sentiment'] == "negative"]  # 9178 rows
    
    random.seed(12345)
    neu_new = shuffle(neu)
    neu_new = neu_new[0:len_pos]  # randomly choose 2363 neutral Tweets
    
    random.seed(123456)
    neg_new = shuffle(neg)
    neg_new = neg_new[0:len_pos]  # randomly choose 2363 negative Tweets
    
    frames = [pos, neu_new, neg_new]
    airline_select = pd.concat(frames)  #concatenate dataframes 
    
    random.seed(1234567)
    airline_select = shuffle(airline_select)  # random Tweets
       
    #print(airline_select.shape)  # (7089, 9)    
    #airline_select.sample(20)
    
    return airline_select

# show world cloud graph
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color = 'white',
        #contour_width = 3, 
        #contour_color = 'steelblue',
        stopwords = stopwords_forCloud,
        max_words = 80,
        max_font_size = 50, 
        scale = 3,
        random_state = 1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    plt.figure(1, figsize = (12, 12))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()

# data exploratory analysis    
def airline_EDA():
    
    # Barplot shows the number neutral, positive and neguative reviews.
    sns.set(style="darkgrid")
    sns.countplot(x = 'airline_sentiment', data = airline_sub, order = airline_sub['airline_sentiment'].value_counts().index, palette = 'Set1')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()
    
    # barplot shows review numbers of each airline, as well as the frequency of sentiment
    sns.set(style="darkgrid")
    sns.countplot(x = 'airline', data = airline_sub, hue = 'airline_sentiment', order = airline_sub['airline'].value_counts().index, palette = 'Set2')
    plt.xlabel('Airline')
    plt.ylabel('Frequency')
    plt.legend().set_title('Sentiment')
    plt.show()
            
    # barplot shows the frequency of the different negative reasons          
    sns.set(style="darkgrid")
    sns.countplot(y = 'negativereason', data = airline_sub, order = airline_sub['negativereason'].value_counts().index, palette = 'Set1')
    plt.xlabel('Frequency')
    plt.ylabel('Negative Reason')
    plt.show()
    
    # barplot shows the distribution of negative reasons on each airlines
    plt.figure(figsize=(12, 6))
    sns.countplot(x = 'airline', data = airline_sub, hue = 'negativereason', palette = 'Set2', saturation = True)
    plt.xlabel('Airline')
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor = (1.01, 1), loc = 2, borderaxespad = 0.1)
    plt.show() 
    
    # boxplot shows the review length distribution over neutral, positive and negative sentiment
    sns.boxplot(x = 'airline_sentiment', y = 'text_length', data = airline_sub)    
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.show()   
    
    # show wourld cloud for negative, neutral and positive tweets
    airline_sub_neg = airline_sub.loc[airline_sub['airline_sentiment'] == 'negative']   # select rows with negative sentiment
    airline_sub_neu = airline_sub.loc[airline_sub['airline_sentiment'] == 'neutral']    # select rows with neutral sentiment    
    airline_sub_pos = airline_sub.loc[airline_sub['airline_sentiment'] == 'positive']   # select rows with positive sentiment

    show_wordcloud(airline_sub_neg['text'])     # show world cloud with negative sentiment
    show_wordcloud(airline_sub_neu['text'])     # show world cloud with neutral sentiment
    show_wordcloud(airline_sub_pos['text'])     # show world cloud with positive sentiment    
 
 # use Wordnet(lexical database) to lemmatize text 
def lemmatize_text(text):
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()    
    text = [lemmatizer.lemmatize(t) for t in text]
    return (' '.join(text))

# clean and normalize text
def pre_process(text):
    
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
    
    text = re.sub(r'@\w+ ?', '', text)        # remove user mentions, e.g. @VirginAmerica 
    text = re.sub(r'#\w+ ?', '', text)        # remove hashtags, e.g. #united
    text = re.sub(r'http\S+', '', text)       # remove URL links  
    text = re.sub(r'\"', '', text)
    
    text = ''.join([t for t in text if not t.isdigit()])   # remove all numeric digits
    
    text = emoji_pattern.sub(r'', text)                    # remove emojis 
    
    text = text.lower()                                                               # lowercase all letters   
    text = ''.join([t for t in text if t not in string.punctuation])                  # remove all punctuations
    #text = ''.join([t for t in text if t not in stopwords.words('english')])  # remove stopwords   
    
    text = lemmatize_text(text)   # use Wordnet(lexical database) to lemmatize text 
    
    return text

##Test one of the messages
#print(pre_process("@united Flight flights find 472 from ORD couldn't lets  123 me know about this? \
#                  Found out via app minutes before landing. Awful flights. #liars #united http://t.co/DQJl8vZ2h2"))
    
# Naive Bayes model
def airline_NB(df, feature, ngram):    
    random.seed(1234567)
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', stop_words="english", ngram_range=(1, ngram))
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(analyzer = 'word', stop_words="english", ngram_range=(1, ngram))
    
    vector_output = vector.fit_transform(df['processed_text'])
    print(vector.get_feature_names())
       
    X_train, X_test, y_train, y_test = train_test_split(vector_output, df['label'], test_size = 0.2, random_state = 101)   
    
    mnb = MultinomialNB(alpha = 0.2)
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    
    confusion_matrix_result = confusion_matrix(y_test, predictions)
    print("NB confusion matrix:", feature)
    print(confusion_matrix_result)

    print("NB classification report:", feature)
    print(classification_report(y_test, predictions))
    

# Logistic regression model    
def airline_LogisticRegression(df, feature):
    random.seed(1234567)
    if feature == "TF":
        vector = CountVectorizer(analyzer = pre_process)
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(analyzer = pre_process)
    vector_output = vector.fit_transform(df['text'])
    
    X_train, X_test, y_train, y_test = train_test_split(vector_output, df['label'], test_size = 0.2, random_state = 101)
    Logistic_model = sklearn.linear_model.LogisticRegression(penalty = "l1", C = 0.1)
    Logistic_model.fit(X_train, y_train)
    
    predictions = Logistic_model.predict(X_test)
    
    confusion_matrix_result = confusion_matrix(y_test, predictions)
    print("Logistic Regression confusion matrix:", feature)
    print(confusion_matrix_result)

    print("Logistic Regression classification report:", feature)
    print(classification_report(y_test, predictions))
    
 # SVM model   
def airline_SVM(df, feature, ngram):
    random.seed(1234567)
    print("ngram", ngram)
    if feature == "TF":
        vector = CountVectorizer(analyzer = pre_process, ngram_range=(1, ngram))
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(analyzer = pre_process, ngram_range=(1, ngram))
    
    vector_output = vector.fit_transform(df['text'])
    
    X_train, X_test, y_train, y_test = train_test_split(vector_output, df['label'], test_size = 0.2, random_state = 101)
    
    SVM_linear = svm.SVC(kernel = 'linear')
    SVM_linear.fit(X_train, y_train)
    predictions = SVM_linear.predict(X_test)
        
    confusion_matrix_result = confusion_matrix(y_test, predictions)
    print("SVM confusion matrix:", feature)
    print(confusion_matrix_result)

    print("SVM classification report:", feature)
    print(classification_report(y_test, predictions))
    


# main function
def main():    
    airline_EDA()
    
    airline_sub['processed_text'] =  airline_sub['text'].apply(pre_process)
     
    print("Process on the whole dataset")    
    airline_NB(airline_sub, "TF", 3)                # Naive Bayes  TF
    airline_NB(airline_sub, "TFIDF", 3)             # Naive Bayes  TF-IDF
    
    airline_LogisticRegression(airline_sub, "TF")         # Logistic Regression  TF
    airline_LogisticRegression(airline_sub, "TFIDF")      # Logistic Regression  TF-IDF

    airline_SVM(airline_sub, "TF", 3)           # SVM  TF
    airline_SVM(airline_sub, "TFIDF", 3)        # SVM  TF-IDF
    

    airline_select = data_resample(airline_sub)       # data selection in order to get a balanced dataset
        
    print(airline_select['processed_text'])
    
    print("Process on the selected and balanced dataset")     
    airline_NB(airline_select, "TF", 3)    
    airline_NB(airline_select, "TFIDF", 3)
    
    airline_LogisticRegression(airline_select, "TF")
    airline_LogisticRegression(airline_select, "TFIDF")
    
#    airline_SVM(airline_select, "TF")
#    airline_SVM(airline_select, "TFIDF")  
    
    print("SVM with trigrams")
    airline_SVM(airline_select, "TF", 3)
    airline_SVM(airline_select, "TFIDF", 3)
    
    
    
if __name__ == "__main__":
    main()


 

     


               