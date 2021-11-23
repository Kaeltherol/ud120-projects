import numpy as np
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

###parameters

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

max_features = ['auto','sqrt']

max_depth = [2,4]

min_sample_split = [2,5]

min_sample_leaf = [1,2]

bootstrap = [True, False]

#create parameter grid

param_grid = {
    'n_estimators' : n_estimators,

    'max_features' : max_features,

    'max_depth' : max_depth,

    'min_samples_split' : min_sample_split,

    'min_samples_leaf' : min_sample_leaf,

    'bootstrap' : bootstrap
}

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

rfGrid = GridSearchCV(estimator=rf_model,param_grid=param_grid, cv=3, verbose = 2, n_jobs=4)

rfGrid.fit(features_train,labels_train)

print(rfGrid.best_params_)

print(rfGrid.best_score_)
print(rfGrid.score(features_test,labels_test))