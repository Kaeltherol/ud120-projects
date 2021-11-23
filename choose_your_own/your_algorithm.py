#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from classifier import classifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
fig, axs = plt.subplots(1,3, figsize=(12, 4), sharey=True)
axs[0].set_xlim(0.0, 1.0)
axs[0].set_ylim(0.0, 1.0)
axs[1].set_xlim(0.0, 1.0)
axs[1].set_ylim(0.0, 1.0)
axs[2].set_xlim(0.0, 1.0)
axs[2].set_ylim(0.0, 1.0)
axs[0].scatter(bumpy_fast, grade_fast, color = "b", label="fast")
axs[0].scatter(grade_slow, bumpy_slow, color = "r", label="slow")
axs[1].scatter(bumpy_fast, grade_fast, color = "b", label="fast")
axs[1].scatter(grade_slow, bumpy_slow, color = "r", label="slow")
axs[2].scatter(bumpy_fast, grade_fast, color = "b", label="fast")
axs[2].scatter(grade_slow, bumpy_slow, color = "r", label="slow")
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[0].set_xlabel("bumpiness")
axs[0].set_ylabel("grade")
axs[1].set_xlabel("bumpiness")
axs[1].set_ylabel("grade")
axs[2].set_xlabel("bumpiness")
axs[2].set_ylabel("grade")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC


CRITERIONS = ['gini','entropy']
choice_1 = 0
METHODS = [RFC(n_estimators =10, criterion='gini', bootstrap=False),
            RFC(n_estimators =10, criterion='gini',bootstrap=False, verbose = 2),
            RFC(n_estimators=100,criterion='gini',min_impurity_decrease=1.0)]

for choice in range(2):
    print(f'\nClassificator: {METHODS[choice]}')
    #print(f'Criterion: {CRITERIONS[choice_1]}')
    print('++Classification in progress, it may take a while..')
    clf, pred_label, acc_score, tot_time = classifier(
        features_train,labels_train,features_test,labels_test,METHODS[choice])

    prettyPicture(clf, features_test, labels_test, axs[choice])


plt.show()
