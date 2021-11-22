#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
import os
import sys
from time import time
os.chdir("C:\\Python\\MyRepo\\ud120-projects\\tools")    
sys.path.append(r"../tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import tree
from sklearn.metrics import accuracy_score

for num_samples_split in [2,25,40]:
    clf = tree.DecisionTreeClassifier(min_samples_split= num_samples_split)

    t0 = time()
    clf.fit(features_train,labels_train)
    fit_time = round(time()-t0,3)

    t1 = time()
    pred = clf.predict(features_test)
    pred_time = round(time()-t1,3)

    acc = round(accuracy_score(labels_test,pred),5)

    print('Classification Summary:\n')
    print('Classificator used: DecisionTreeClassifier')
    print(f'Minimum Samples Split: {num_samples_split}')
    print(f'Accuracy: {acc}')
    print(f'Fit time: {fit_time} s')
    print(f'Pred time: {pred_time} s')
    print(f'Total time: {fit_time+pred_time} s')






#########################################################
### your code goes here ###


#########################################################


