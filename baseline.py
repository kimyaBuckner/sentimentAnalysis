import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re, string
import os


#Create the bag of words
relevantWords = []
totalReviews = []

stop_words = list(set(stopwords.words('english')))

neg_data_files = os.listdir('neg')
original_neg_data = [open('neg/'+ fileName, 'r').read() for fileName in neg_data_files]

pos_data_files = os.listdir('pos')
original_pos_data = [open('pos/'+ fileName, 'r').read() for fileName in pos_data_files]


#Gather negative words 
for review in original_neg_data:

    totalReviews.append((review, "n"))

    punctuationless_review = re.sub(r'[^(a-zA-Z)\s]','', review)

    tokenized_review = word_tokenize(punctuationless_review)

    stoppless_review = [word for word in tokenized_review if not word in stop_words]

    pos_tagged_review = nltk.pos_tag(stoppless_review)

    for word in pos_tagged_review:
         if(word[1] == "JJ"):
            relevantWords.append(word[0])

# Gather positive words
for review in original_pos_data:

    totalReviews.append((review, "p"))

    punctuationless_review = re.sub(r'[^(a-zA-Z)\s]','', review)

    tokenized_review = word_tokenize(punctuationless_review)

    stoppless_review = [word for word in tokenized_review if not word in stop_words]

    pos_tagged_review = nltk.pos_tag(stoppless_review)

    for word in pos_tagged_review:
         if(word[1] == "JJ"):
            relevantWords.append(word[0])

print(len(totalReviews))

bag_of_words = nltk.FreqDist(relevantWords)

word_features = list(bag_of_words.keys())[:5000]


#for every word in our bag of words check to see if that word is in the review. Set word to true or false depending
def find_features(review):
    words = word_tokenize(review)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

featuresets = [(find_features(review), sentiment) for (review, sentiment) in totalReviews ]

random.shuffle(featuresets)

training_set = featuresets[:1500]
testing_set = featuresets[1500:]


classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)