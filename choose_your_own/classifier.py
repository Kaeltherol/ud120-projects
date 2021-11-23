
def classifier(features_train,labels_train,features_test,labels_test,chosen_classifier):

    clf = chosen_classifier
    from time import time
    t0 = time()
    clf.fit(features_train,labels_train)
    fit_time = round(time()-t0,3)

    t1 = time()
    pred = clf.predict(features_test)
    pred_time = round(time()-t1,3)
    total_time = round(fit_time+pred_time,3)
    from sklearn.metrics import accuracy_score
    acc = round(accuracy_score(labels_test,pred),5)

    print('\nClassification Summary:\n')
    print(f'Accuracy: {acc}')
    print(f'Fit time: {fit_time} s')
    print(f'Pred time: {pred_time} s')
    print(f'Total time: {total_time} s\n')

    return clf, pred, acc, total_time
