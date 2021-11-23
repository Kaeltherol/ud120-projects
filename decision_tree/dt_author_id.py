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
from classifier import classifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print(len(features_train[0]))

from sklearn import tree

for num_samples_split in [40]:

    print('\nClassificator: Decision Tree Classifier')
    print(f'Min Sample Split: {num_samples_split}')
    print('++Classification in progress, it may take a while..')
    pred_label, acc_score, tot_time = classifier(
        features_train,labels_train,features_test,labels_test,tree.DecisionTreeClassifier(min_samples_split= num_samples_split))






#########################################################
### your code goes here ###


#########################################################


