# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:06:49 2019

@author: Rayan Suryadikara / s2432234
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def text_classification(prep, clas, tr_data, ts_data, tr_target, ts_target, twe):
    
    if prep==1:     
        vect = CountVectorizer()
        if twe:
            vect = CountVectorizer(lowercase=False, stop_words='english', analyzer='char', ngram_range=(1,3), max_features=10000)
    elif prep==2:
        vect = TfidfVectorizer(use_idf=False)
        if twe:
            vect = TfidfVectorizer(use_idf=False, lowercase=False, stop_words='english', analyzer='char', ngram_range=(1,3), max_features=10000)
    elif prep==3:
        vect = TfidfVectorizer()
        if twe:
            vect = TfidfVectorizer(lowercase=False, stop_words='english', analyzer='char', ngram_range=(1,3), max_features=10000)
        
    train_vect = vect.fit_transform(tr_data)
    test_vect = vect.transform(ts_data)
    
    if clas==1:     
        clf = MultinomialNB().fit(train_vect, tr_target)
    elif clas==2:
        clf = SGDClassifier().fit(train_vect, tr_target)
    elif clas==3:
        clf = KNeighborsClassifier().fit(train_vect, tr_target)

    prd = clf.predict(test_vect)
    return prd
        
twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

listpre = ["Count Vectorizer", "TF Vectorizer", "TF-IDF Vectorizer"]
listcls = ["Naive Bayes", "Stochastic Gradient Descent", "K-Nearest Neighbors"]

while True:
    print("Preprocessors:")
    print("1. Count Vectorizer")
    print("2. TF Vectorizer")
    print("3. TF-IDF Vectorizer")
    preinp = int(input("Please choose preprocessors based on the number: "))
    if preinp >= 1 and preinp <= 3:
        break;
    else:
        print("Wrong number! Try again.")

while True:
    print("Classifiers:")
    print("1. Naive Bayes")
    print("2. Stochastic Gradient Descent")
    print("3. K-Neighbors")
    clsinp = int(input("Please choose classifiers based on the number: "))
    if clsinp >= 1 and clsinp <= 3:
        break;
    else:
        print("Wrong number! Try again.")

while True:
    tweinp = input("Please type Y or N for tweaking the vectorizer parameters: ")
    if tweinp == 'Y' or tweinp == 'y':
        tweinp = True
        break;
    elif tweinp == 'N' or tweinp == 'n':
        tweinp = False
        break;
    else:
        print("Wrong letter! Try again.")

pred = text_classification(preinp, clsinp, 
                    twenty_train.data, twenty_test.data,
                    twenty_train.target, twenty_test.target, tweinp)

if tweinp:
    tweinp = "with"
else:
    tweinp = "without"
print("Classification Report for", listpre[preinp-1], "&", listcls[clsinp-1], "%s tweaking."%tweinp)
print(classification_report(twenty_test.target, pred, target_names=twenty_test.target_names))


