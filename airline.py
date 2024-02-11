# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:16:30 2019

@author: Yaya Liu
@date: 2019/9/20

1. Introduction of this project
The main goals of this project are to perform sentiment analysis in the area of US airline service 
using Twitter data and explore techniques that  are related to sentiment analysis.

Sentiment analysis means inspecting the given Tweet and determining a user’s attitude as positive, 
negative, or neutral.

Input: Tweets.csv -> the original data
       glove.twitter.27B.100d.txt -> GloVe pre-trained word representation vectors
       
Output: train.csv and test.csv  -> training and testing data for LSTM networs which run on Google Colab
        csv files including predictions made by each model.
        
2. Workflow of this project
    - Import the original data.
    - Data exploratory analysis including visualizations.
    - Tweets preprocessing.
    - Naive Bayes and SVM with different datasets(the original dataset, oversampling data, undersampling data) 
      and different feature creation techniques (TF-IDF with bag-of-words, TF-IDF with the combination of bag-of-words,
      bigrams and trigrams, GloVe pre-trained model).
    - model classification report, generate confusion matrix, save model predictions for error analysis.
"""

import random
import pandas as pd
import numpy as np
from statistics import mean
from scipy import spatial
import re
import string
import itertools

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
#
#from imblearn.over_sampling import SMOTE
#import imblearn.pipeline as smpipe


from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk import PorterStemmer as Stemmer
from nltk import pos_tag

import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
#print(len(stopWords))
#print(stopWords)

from wordcloud import WordCloud, STOPWORDS
stopwords_forCloud = set(STOPWORDS)
stopwords_forCloud.update(['flight', 'flights', 'Flightled', 'AmericanAir', 'VirginAmerica'])


import matplotlib.pyplot as plt
import seaborn as sns
import codecs

np.random.seed(500) 

airline = pd.read_csv('Tweets.csv', encoding = 'utf8')
airline.shape              # (rows: 14640, columns:15)
#print(airline.head(5))

# subset the colums that will be used for analysis
airline_sub = airline.loc[:, ['airline_sentiment', 'airline', 'name', 'negativereason', 'text']]
airline_sub.isnull().sum()    # checking missing values of each column  
#print(airline_sub.shape)   # (rows: 14640, columns: 7)
#print(airline_sub['text'].head(100))
    
sentiment_count = airline_sub['airline_sentiment'].value_counts()  # negative 9178, neutral 3099, positive 2363
print("airline_sub sentiment_count: ", sentiment_count)

airline_sub['label'] = airline_sub['airline_sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1}) # Negative: -1, neutral: 0, positive 1

# label for doc2vec, because it cannpt take -1 as the tag
airline_sub['label_doc2vec'] = airline_sub['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}) # Negative: 0, neutral: 1, positive 2

#print(airline_sub)

airline_sub['text_length'] = airline_sub['text'].apply(lambda x: len(word_tokenize(x)))             # add a column to store the length of each tweet
#airline_sub.head(5)   
    
# show word cloud graph
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

    plt.figure(1, figsize = (10, 12))
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
    plt.xlabel('Airline companies')
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor = (1.01, 1), loc = 2, borderaxespad = 0.1)
    plt.show() 
    
    # boxplot shows the review length distribution over neutral, positive and negative sentiment
    sns.boxplot(x = 'airline_sentiment', y = 'text_length', data = airline_sub)    
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.ylim(0, 50)
    plt.show()   
    
    # show word cloud for negative, neutral and positive tweets
    airline_sub_neg = airline_sub.loc[airline_sub['airline_sentiment'] == 'negative']   # select rows with negative sentiment
    airline_sub_neu = airline_sub.loc[airline_sub['airline_sentiment'] == 'neutral']    # select rows with neutral sentiment    
    airline_sub_pos = airline_sub.loc[airline_sub['airline_sentiment'] == 'positive']   # select rows with positive sentiment

    show_wordcloud(airline_sub_neg['text'])     # show word cloud with negative sentiment
    show_wordcloud(airline_sub_neu['text'])     # show word cloud with neutral sentiment
    show_wordcloud(airline_sub_pos['text'])     # show word cloud with positive sentiment    
 
def stemmer_text(text):
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    st = Stemmer()  
    text = [st.stem(t) for t in text]
    return (' '.join(text))
    
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
 # use Wordnet(lexical database) to lemmatize text 
def lemmatize_text(text):
    
    lmtzr = WordNetLemmatizer().lemmatize
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    word_pos = pos_tag(text)    
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
    return (' '.join(lemm_words))

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

    text = emoji_pattern.sub(r'', text)                       # remove emojis       
    text = text.lower()                                       # lowercase all letters   
#    text = re.sub(r'@[A-Za-z0-9]+', '', text)                # remove user mentions, e.g. @VirginAmerica    
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)       # remove URL links 

#    white_list = ["not", "no", "won't", "isn't", "couldn't", "wasn't", "didn't", "shouldn't", 
#                  "hasn't", "wouldn't", "haven't", "weren't", "hadn't", "shan't", "doesn't",
#                  "mightn't", "mustn't", "needn't", "don't", "aren't", "won't"]
#    words = text.split()
#    text = ' '.join([t for t in words if (t not in stopwords.words('english') or t in white_list)])  # remove stopwords        

    text = ''.join([t for t in text if t not in string.punctuation])   # remove all punctuations       
    text = ''.join([t for t in text if not t.isdigit()])   # remove all numeric digits     
    text = re.sub("[^a-zA-Z0-9]", " ", text)   # letters only         
    text = lemmatize_text(text)   # use Wordnet(lexical database) to lemmatize text     
#    text = stemmer_text(text)   # stem text 
    return text

##Test one of the messages
#print(pre_process("really missed a prime opportunity for men without hats parody"))
    
    
def important_features(vectorizer, classifier, n):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()

    topn_negative = sorted(zip(classifier.feature_count_[0], feature_names), reverse = True)[:n]
    topn_neutral = sorted(zip(classifier.feature_count_[1], feature_names), reverse = True)[:n]
    topn_positive = sorted(zip(classifier.feature_count_[2], feature_names), reverse = True)[:n]

    #print(classifier.feature_count_)
    
    print("-----------------------------------------")
    print("Important features in negative reviews")

    for coef, feat in topn_negative:
        print(class_labels[-1], coef, feat)

    print("-----------------------------------------")
    print("Important features in neutral reviews")

    for coef, feat in topn_neutral:
        print(class_labels[0], coef, feat)
        
    print("-----------------------------------------")
    print("Important features in positive reviews")

    for coef, feat in topn_positive:
        print(class_labels[1], coef, feat)


# perform oversampling on train dataset
def oversampling(train_X):
    df_major_neg = train_X[train_X['label'] == -1]
    df_minor_neu = train_X[train_X['label'] == 0]
    df_minor_pos = train_X[train_X['label'] == 1]        
    major_count = len(df_major_neg)
 
    # oversample minority class
    df_minor_neu_oversampled = resample(df_minor_neu, 
                                 replace = True,              # sample with replacement
                                 n_samples = major_count,     # to match majority class 
                                 random_state = 1000)    

    df_minor_pos_oversampled = resample(df_minor_pos, 
                                 replace = True,             
                                 n_samples = major_count,   
                                 random_state = 1000)      
         
    trainX = pd.concat([df_major_neg, df_minor_neu_oversampled, df_minor_pos_oversampled])   # Combine majority class with oversampled minority class
    print("Train dataset calss distribution: \n", trainX.label.value_counts())
    trainX = shuffle(trainX, random_state = 200) 
    return trainX

def undersampling(train_X):
    df_major_neg = train_X[train_X['label'] == -1]
    df_minor_neu = train_X[train_X['label'] == 0]
    df_minor_pos = train_X[train_X['label'] == 1]        
    minor_count = len(df_minor_pos)
 
    # undersample minority class
    df_major_neg_undersampled = resample(df_major_neg, 
                                 replace = True,              # sample with replacement
                                 n_samples = minor_count,     # to match minority class
                                 random_state = 1000)    

    df_minor_neu_undersampled = resample(df_minor_neu, 
                                 replace = True,             
                                 n_samples = minor_count,   
                                 random_state = 1000)      
         
    trainX = pd.concat([df_major_neg_undersampled, df_minor_neu_undersampled, df_minor_pos])   # Combine majority class with oversampled minority class
    print("Train dataset calss distribution: \n", trainX.label.value_counts())
    trainX = shuffle(trainX, random_state = 200) 
    return trainX
        
# Naive Bayes model
def airline_NB(df, feature, ngram, sample_method):    
    random.seed(1234567)
        
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', ngram_range=(1, ngram)) 
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(analyzer = 'word', ngram_range=(1, ngram))
    #vector_output = vector.fit_transform(df['processed_text'])   
    #print(vector.get_feature_names())      
    #print(df[['processed_text', 'label']])  

    
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)
    
        
    train_X.to_csv("train.csv")
    test_X.to_csv("test.csv")
        
    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)   
              
    pipe = make_pipeline(vector, MultinomialNB(alpha = 1.0, fit_prior = True))
    clf = pipe.fit(train_X['processed_text'], train_X['label'])     
    #print(test_y.value_counts())  # -1:1817, 0: 628, 1:483        
    #print(vector.get_feature_names())
    
    test_y_hat = pipe.predict(test_X['processed_text'])
    #test_y_hat_prob = pipe.predict_proba(test_X['processed_text'])
    
    important_features(vector, clf[1], 20)
    
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_X['processed_text']), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)
    
    file_name = 'NB_' + str(ngram) + '_' + sample_method 
    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    print("NB classification report -- ", "feature: %s/" %feature, "ngram: %d/" %ngram, "sample_method: %s/" %sample_method)
    print(pd.crosstab(test_y.ravel(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))      

    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))    
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))      

# Naive Bayes model
def airline_SVM(df, feature, ngram, sample_method):    
    random.seed(1234567)
        
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', ngram_range=(1, ngram)) 
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(analyzer = 'word', ngram_range=(1, ngram))
    #vector_output = vector.fit_transform(df['processed_text'])   
    #print(vector.get_feature_names())      
    #print(df[['processed_text', 'label']])    
    
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)

    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)               
 
    pipe = make_pipeline(vector, svm.SVC(kernel = 'linear', probability = True, random_state = 101))
    clf = pipe.fit(train_X['processed_text'], train_X['label'])     
    #print(train_y.value_counts())  # -1:7361, 0: 2471, 1:1880
    #print(test_y.value_counts())  # -1:1817, 0: 628, 1:483        
    #print(vector.get_feature_names())
    
    test_y_hat = pipe.predict(test_X['processed_text'])
    #test_y_hat_prob = pipe.predict_proba(test_X['processed_text'])
    
    #important_features(vector, clf[1], 20)  # SVM does not support this function
        
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_X['processed_text']), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)
    
    file_name = 'SVM_' + str(ngram) + '_' + sample_method 
    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    print("SVM classification report -- ", "feature: %s/" %feature, "ngram: %d/" %ngram, "sample_method: %s/" %sample_method)
    print(pd.crosstab(test_y.ravel(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))  
    
    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))
    
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))  

# load pre-trained Tweets model
# https://nlp.stanford.edu/projects/glove/
def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """
    print("Loading Glove Model")
    f = open(glove_file,'r', encoding = "utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model  


count_total = 0   # How many words are in processed data, including duplicate words
count_in = 0      # How many words are in Glove pretrained data
count_out = 0     # How many words are not in Glove pretrained data
out_words_list = []    # words list that not in Glove pretrained data

# get vector for each word, add vectors and take the average of the vector
def tweet_to_vec(tweet, g_model, num_features):    
    
    global count_total, count_in, count_out
    
    word_count = 0
    feature_vec = np.zeros((num_features), dtype = "float32")
    
    for word in tweet.split(' '):
        count_total += 1
        if word in g_model.keys():   
            count_in += 1
            word_count += 1
            feature_vec += g_model[word]
        else:
            count_out += 1
            out_words_list.append(word)
    if (word_count != 0):
        feature_vec /= word_count
    return feature_vec

# get word2vec vector for each tweet        
def gen_tweet_vecs(tweets, g_model, num_features):    
    curr_index = 0
    tweet_feature_vecs = np.zeros((len(tweets), num_features), dtype = "float32")
    
    for tweet in tweets:
        if curr_index % 2000 == 0:
            print('Word2vec vectorizing tweet %d of %d' %(curr_index, len(tweets)))
        tweet_feature_vecs[curr_index] = tweet_to_vec(tweet, g_model, num_features)
        curr_index += 1
    return tweet_feature_vecs   
 
# visualize word2vec model and Tweets distribution after using PCA on word2vec embeddings    
def airline_word2vec_visualization(df, g_model):
    
    # model on words
    words = ['thank', 'great', 'best', 'love', 'cool', 'appreciate', 'awesome', 'like', 'deal', 'good', \
             'delay', 'cancel', 'cant', 'hold', 'not', 'no', 'bad', 'terrible', 'wait', 'miss', 'but', \
             'can', 'do', 'fleet', 'flight', 'roundtrip', 'change', 'travel', 'possible',  'customer', 'service']
    
    words_vec = []
    for word in words:
        words_vec.append(g_model[word])

    words_vec = StandardScaler().fit_transform(words_vec)
        
    pca = PCA(n_components = 2)
    result = pca.fit_transform(words_vec)
    
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2-component PCA on words', fontsize = 20)
        
    for i, word in enumerate(words):
        ax.annotate(word, xy=(result[i, 0], result[i, 1]))
        ax.scatter(result[i, 0], result[i, 1])
    

   # model on classification   
    x = df['processed_text']
    y = df['label'].values
    x_vec = gen_tweet_vecs(x, g_model, 100)
    x_vec = StandardScaler().fit_transform(x_vec)
    
    pca = PCA(n_components=2)
    pca = pca.fit_transform(x_vec)
    
    explained_variance = np.var(pca, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    print("PCA: explained variance")
    print(explained_variance_ratio)

    print("PCA: explained variance cumsum:")
    print(np.cumsum(explained_variance_ratio)) 
    
    principalDf = pd.DataFrame(data = pca, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2-component PCA on All Data', fontsize = 20)
    targets = [-1, 0, 1]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)  
    
    
# build models based on word2vec embeddings    
def airline_word2vec_model(df, classifier, g_model, sample_method):
      
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)   

    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)         

    global count_total, count_in, count_out
    global out_words_list
    count_total, count_in, count_out = 0, 0, 0 
    out_words_list = []    
    
    train_vec = gen_tweet_vecs(train_X['processed_text'], g_model, 100)
    test_vec = gen_tweet_vecs(test_X['processed_text'], g_model, 100)
    
    print("Glove word embedding statistic\n", "count_total: %d/" %count_total, "count_in: %d/" %count_in, "count_out: %d/" %count_out)
    print("Number of unique words without embedding: %d" %len(set(out_words_list)))
    print("Words without embedding: \n", set(out_words_list))

    
#    train_vec = StandardScaler().fit_transform(train_vec)
#    test_vec = StandardScaler().fit_transform(test_vec)
    
    if classifier == "SVM":      
        pipe = make_pipeline(svm.SVC(kernel = 'linear', probability = True, random_state = 101))
        clf = pipe.fit(train_vec, train_X['label'])                 
        test_y_hat = pipe.predict(test_vec)
        file_name = 'SVM_word2vec_' + sample_method                  
    elif classifier == "RF":             
        pipe = make_pipeline(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0))
        clf = pipe.fit(train_vec, train_X['label'])                 
        test_y_hat = pipe.predict(test_vec)            
        file_name = 'RF_word2vec_' + sample_method  
            
                        
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_vec), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)

    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    if classifier == "SVM": 
        print("SVM word2vec classification report -- ", "sample_method: %s" %sample_method) 
    elif classifier == "RF":
        print("RF word2vec classification report -- ", "sample_method: %s" %sample_method) 
        
    print(pd.crosstab(test_y.ravel(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))       
    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))

    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))  
        

# train docutments for training and test
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


# build models based on doc2vec embeddings    
def airline_doc2vec_model(df, classifier, sample_method):  
        
    train_X, test_X, train_y, test_y = train_test_split(df, df['label_doc2vec'], test_size = 0.2, random_state = 101)       
    
    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)  

    train_X_tagged = train_X.apply(
        lambda r: TaggedDocument(words = word_tokenize(r['processed_text']), tags=[r.label_doc2vec]), axis=1)
    test_X_tagged = test_X.apply(
        lambda r: TaggedDocument(words = word_tokenize(r['processed_text']), tags=[r.label_doc2vec]), axis=1)
    
    #print(train_X_tagged.values[30])
    
    doc2vec_model = Doc2Vec(train_X_tagged, dm = 0, vector_size = 300, min_count = 1, epochs = 100)        
    doc2vec_model_test = Doc2Vec(test_X_tagged, dm = 0, vector_size = 300, min_count = 1, epochs = 100)
    
#    doc2vec_model.save("doc2vec.model")
    
    # test sentence similarity based on the doc2vec model    
    s1 = "flt crew to is the weather delay but pilot just invited the kid to see the cockpit"
    s2 = "flt crew to is the weather delay pilot just invited the kid to see the cockpit"
    vec1 = doc2vec_model_test.infer_vector(s1.split())
    vec2 = doc2vec_model_test.infer_vector(s2.split())
    similairty = spatial.distance.cosine(vec1, vec2)
    print("sentence similarity: %f" %similairty)
    
    s1 = "flt crew to is the weather delay but pilot just invited the kid to see the cockpit"
    s2 = "i hope so too thank you for your help she traveled halfway across the globe and just want her suitcase"        
    vec1 = doc2vec_model_test.infer_vector(s1.split())
    vec2 = doc2vec_model_test.infer_vector(s2.split())
    similairty = spatial.distance.cosine(vec1, vec2)
    print("sentence similarity: %f" %similairty)
        
    train_y, train_vec = vec_for_learning(doc2vec_model, train_X_tagged)
    test_y, test_vec = vec_for_learning(doc2vec_model, test_X_tagged)
    
#    logreg = LogisticRegression(n_jobs=1, C=1e5)
#    logreg.fit(X_train, y_train)
#    y_pred = logreg.predict(X_test)
#    from sklearn.metrics import accuracy_score, f1_score
#    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
#    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
    
    if classifier == "SVM":      
        pipe = make_pipeline(svm.SVC(kernel = 'linear', probability = True, random_state = 101))
        clf = pipe.fit(train_vec, train_y)                 
        test_y_hat = pipe.predict(test_vec)
        file_name = 'SVM_doc2vec_' + sample_method                  
    elif classifier == "RF":
        pipe = make_pipeline(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0))
        clf = pipe.fit(train_vec, train_y)                 
        test_y_hat = pipe.predict(test_vec)            
        file_name = 'RF_doc2vec' + sample_method
            
                        
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_vec), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)

    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    if classifier == "SVM": 
        print("SVM doc2vec classification report -- ", "sample_method: %s" %sample_method) 
    elif classifier == "RF":
        print("RF doc2vec classification report -- ", "sample_method: %s" %sample_method)

    print(confusion_matrix(test_y, test_y_hat))  
    
    print("-----------------------------------------")
#    print(pd.crosstab(test_y, test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True)) 
    print(classification_report(test_y, test_y_hat))   
     
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))  


    
# main function
def main():    
#    airline_EDA()    
    
    airline_sub['processed_text'] =  airline_sub['text'].apply(pre_process)  
    airline_sub['processed_text_length'] = airline_sub['processed_text'].apply(lambda x: len(word_tokenize(x)))
            
    # boxplot shows the review length distribution over neutral, positive and negative sentiment
    sns.set(style="darkgrid")
    sns.boxplot(x = 'airline_sentiment', y = 'processed_text_length', data = airline_sub)    
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.ylim(0, 50)
    plt.show() 
    
    
    airline_model = airline_sub.loc[:, ['airline_sentiment', 'text', 'processed_text', 'label', 'label_doc2vec']]
    airline_model.to_csv('airline_model.csv')  
            
    # Naive Bayes. Arguments: dataframe, TF/IFIDF, unigran or ngram, data-balancing method   
    airline_NB(airline_model, "TFIDF", 1, "none")  
    airline_NB(airline_model, "TFIDF", 1, "oversampling")  
    airline_NB(airline_model, "TFIDF", 1, "undersampling")  
            
    airline_NB(airline_model, "TFIDF", 3, "none")
    airline_NB(airline_model, "TFIDF", 3, "oversampling")
    airline_NB(airline_model, "TFIDF", 3, "undersampling")  
    
    
    # SVM. Arguments: dataframe, TF/IFIDF, unigran or ngram, data-balancing method       
    airline_SVM(airline_model, "TFIDF", 1, "none")  
    airline_SVM(airline_model, "TFIDF", 1, "oversampling")  
    airline_SVM(airline_model, "TFIDF", 1, "undersampling")  
      

    airline_SVM(airline_model, "TFIDF", 3, "none") 
    airline_SVM(airline_model, "TFIDF", 3, "oversampling") 
    airline_SVM(airline_model, "TFIDF", 3, "undersampling") 

    g_model = load_glove_model("glove.twitter.27B.100d.txt")    
 
    airline_word2vec_visualization(airline_model, g_model)

    airline_word2vec_model(airline_model, "SVM", g_model, "none")       
    airline_word2vec_model(airline_model, "SVM", g_model, "oversampling")
    airline_word2vec_model(airline_model, "SVM", g_model, "undersampling") 

    airline_word2vec_model(airline_model, "RF", g_model, "oversampling")
    airline_word2vec_model(airline_model, "RF", g_model, "undersampling")
    
    airline_doc2vec_model(airline_model, "SVM", "None")
    airline_doc2vec_model(airline_model, "SVM", "oversampling")
    airline_doc2vec_model(airline_model, "SVM", "undersampling")

    airline_doc2vec_model(airline_model, "RF", "oversampling")
    airline_doc2vec_model(airline_model, "RF", "None")



if __name__ == "__main__":
    main() 


 

     


               
