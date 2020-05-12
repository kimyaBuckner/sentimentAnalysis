import os
import re
import nltk
import string
import random
from nltk.text import Text
from statistics import mode
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

relevantWords = []
totalReviews = []

def getRelevantWords():

    global relevantWords
    global totalReviews
    stop_words = list(set(stopwords.words('english')))
    rePunctuation = r'[^(a-zA-Z)\s]'

    #Read in all the lines for the negative reviews
    neg_data_files = os.listdir('neg')
    original_neg_data = [open('neg/'+ fileName, 'r').read() for fileName in neg_data_files]

     #Read in all the lines for the positive reviews
    pos_data_files = os.listdir('pos')
    original_pos_data = [open('pos/'+ fileName, 'r').read() for fileName in pos_data_files]

    #Preprocess the negative words: strip out punctuation and all stop words. Add all adjetives to the list of relevant words
    for review in original_neg_data:

        totalReviews.append((review, "N"))

        punctuationless_review = re.sub(rePunctuation,'', review)

        tokenized_review = word_tokenize(punctuationless_review)

        stoppless_review = [word for word in tokenized_review if not word in stop_words]

        pos_tagged_review = nltk.pos_tag(stoppless_review)

        for word in pos_tagged_review:
            pos = word[1]
            if( pos == "JJ" or pos == "JJR" or pos == "JJS"  or pos == "RB" or pos == "RBR" or pos == "RBS" ):
                relevantWords.append(word[0])

    #Preprocess the positive words: strip out punctuation and all stop words. Add all adjetives to the list of relevant words
    for review in original_pos_data:

        totalReviews.append((review, "P"))

        punctuationless_review = re.sub(rePunctuation,'', review)

        tokenized_review = word_tokenize(punctuationless_review)

        stoppless_review = [word for word in tokenized_review if not word in stop_words]

        pos_tagged_review = nltk.pos_tag(stoppless_review)

        #Only consider adjectives and adverbs
        for word in pos_tagged_review:
            pos = word [1]
            if( pos == "JJ" or pos == "JJR" or pos == "JJS"  or pos == "RB" or pos == "RBR" or pos == "RBS" ):
                relevantWords.append(word[0])


#for every word in our bag of words check to see if that word is in the review. Set word to true or false depending
def find_features(review, word_features):
    words = word_tokenize(review)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

#Given a classifier, calculate the accuracy, fscore and recall of that classifier
def runClassifier(classifier, classifierType, testing_set):

    actual_sentiment =  [review[1] for review in testing_set]
    predicted_sentimet = [classifier.classify(review[0]) for review in testing_set]

    accuracy = accuracy_score(actual_sentiment, predicted_sentimet, normalize=True)*100
    fscore = f1_score(actual_sentiment, predicted_sentimet, average = "macro")
    recall = recall_score(actual_sentiment, predicted_sentimet, average = "macro")
    precision = precision_score(actual_sentiment, predicted_sentimet, average = "macro")
    print(classifierType + "accuracy percent is: ", accuracy)
    print(classifierType + "fscore is: ", fscore)
    print(classifierType + "recall is ", recall)
    print(classifierType + "recall is ", precision)
    

    print('\n\n')
if __name__ == "__main__":
    print('\nThis model tests how the size of feature sets influences performance. \n This will take a several minutes to run to completion. \n It will take longer than the baseline model. \n It has to train and test six feature sets on 3 machine learning models.\n', flush=True)
    print('\n Please refer to vectorSizingResults.png to see what an example of the completed program produces to std out\n', flush=True)
    getRelevantWords()

    #Get the frequency distributuion of the words to be used in the feature sets
    bag_of_words = nltk.FreqDist(relevantWords)


    print("------------------------------------------------------------------------\n", flush=True)     
    for i in range(5, 11):
        num_words = i * 1000
        print("Now beggining creation of feature set of size ", num_words, flush=True)
        word_features = list(bag_of_words.keys())[:num_words]
        featuresets = [(find_features(review, word_features), sentiment) for (review, sentiment) in totalReviews ]
        
        
        print('\nCreating testing and training sets.\n', flush=True)
        random.shuffle(featuresets)
        
        training_set = featuresets[:1500]
        testing_set = featuresets[1500:]


        logRegressionClassifier = SklearnClassifier(LogisticRegression()).train(training_set)

        print("Now testing on Naive Bayes Classifier.... ", flush=True)
        naiveBayesClassifier = nltk.NaiveBayesClassifier.train(training_set)
        runClassifier(naiveBayesClassifier, "Naive Bayes Classifier ", testing_set)


        print("Now testing on Logistic Regression Classifier.... ", flush=True)
        logRegressionClassifier = SklearnClassifier(LogisticRegression()).train(training_set)
        runClassifier(logRegressionClassifier, "Logistic Regression Classifier ", testing_set)


        print("Now testing on Support Vector Classifier.... ", flush=True)               
        supportVectorClassifier = SklearnClassifier(SVC()).train(training_set)
        runClassifier(supportVectorClassifier, "Support Vector Classifier ", testing_set)

        print("------------------------------------------------------------------------ \n", flush=True)     
