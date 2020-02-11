# US_Airline_Sentiment_Analysis_using_Twitter_Data

1. Introduction

Social media has grown massively in recent years. Each day, around 500 million Tweets are tweeted on Twitter. People share their genuine emotions, feelings, opinions and experiences on social media. More and more companies start using social media data to improve customer service, enhance business competitiveness, perform crisis management, etc.

One of the popular text analysis techniques is sentiment analysis, which is the process of determining whether the writer’s attitude is positive, negative or neutral towards a specific topic, brand, product, company, etc. Comparing to traditional paper-based surveys, sentiment analysis based on online data is more time-saving, economical and more reliable because its ability to access and analyze enormous data. 

Sentiment analysis started in early 2000s. Multiple approaches have been developed and a lot of research has been done in various fields afterwards. But only a few studies directly focused on the area of airlines based on Twitter data. Hence, the main purpose of this project is to provide sentiment classification in the area of US airline service using Twitter data. To be more specific, inspecting the given Tweet and identifying the prevailing emotional opinion within the Tweet, especially to determine a user’s attitude as positive, negative, or neutral.


2. Challenges

The challenges of this project are associated with Twitter data. 

First, the service-related data from Twitter is often unbalanced. There are way more negative Tweets than positive and neutral Tweets, which makes sense because people who has an awful experience is more likely to share that experience on Twitter. The imbalanced data might have an impact on sentiment classification. 

Second, Tweets are often very short. They may not include enough context to decipher things like irony, sarcasm, etc. 

Third, unlike formal publications, Tweets often have a lot of noise comparing to published articles, such as emojis, emoticons, external links, user mention, a lot of white spaces, etc. Tweets require extra text clean before feeding them to machine learning models. 

In addition, since most of the Tweets are very short, it is essential to identify which type of noise should be kept. 


3. Methods    

Naïve Bayes model, random forest, support vector machine (SVM), and long short-term memory networks (LSTM networks) were used to conduct sentiment analysis. 

In order to evaluate the impact caused by the imbalanced data, undersampling and oversampling techniques were used to balanced data.

For feature creation, TF-IDF (frequency–inverse document frequency) with bag-of-words, TF-IDF with the combination of bag-of-words, bigrams and trigrams were used to create features. Since these two methods are based on bag-of-words, they do not preserve the order of the words in each Tweet. It is unlikely that they can capture the prevailing context of Tweets. Therefore, GloVe pre-trained word representation vectors (100-dimensional) were used to represent each word. For a Tweet, I added up the vector of each word in this Tweet and took the average of the sum to represent this Tweet. In addition, Doc2Vec was also used to embed each Tweet.

For model evaluation, micro-average recall, precision, F1 score were used to measure model performance. The micro-average is the average score for each class. 

Micro-average Recall = (Recallclass1 + Recallclass2 + Recallclass3) / 3

Micro-average Precision = (Precisionclass1 + Precisionclass2 + Precisionclass3) / 3

Micro-average F1 score = (F1class1 + F1class2 + F1class3) / 3

For each class, recall, precision and F1 score can be calculated like binary classification. In binary classification model, recall is the ratio of correctly classified instances for one class of overall instances in this class. Precision is the fraction of the correctly classified instances for one class of the overall instances which are classified to this class. F1 score is the harmonic mean of the precision and recall. It keeps a balance between precision and recall and gets a comprehensive evaluation of the models.

4. Data  

Data is from Kaggle. It includes 14, 640 Tweets covering six U.S. airline companies. Each Tweet has already been labeled as “negative”, “neutral” or “positive” class. Among these Tweets, 9, 178 Tweets are negative, 3, 099 Tweets are neutral, and 2, 363 Tweets are positive. 

Data link: https://www.kaggle.com/welkin10/airline-sentiment

5. Results
Please check file "US Airline sentiment analysis using Twitter data .pdf"


