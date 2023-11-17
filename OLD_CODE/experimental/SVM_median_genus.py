from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as scl
from sklearn import svm, grid_search
from LoadData import LoadData
from LoadData import TribeLoader
from LoadData import GenusLoader
from sklearn.model_selection import train_test_split #being deprecated, works for now
from sklearn.model_selection import GridSearchCV #being deprecated, works for now
from sklearn.metrics import classification_report
from sklearn.svm import SVC

np.random.seed(13)
 
gl = GenusLoader(32) # your images will be size * size
train_images, train_labels, train_tribe = gl.load('/Users/jessicaperry/Dropbox/phytoliths/median_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/genus_train.txt')
    #"images" are (num_of_images, size, size) shaped np array, "labels" are (num_of_images,) shaped np array
test_images, test_labels, test_tribe = gl.load('/Users/jessicaperry/Dropbox/phytoliths/median_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/genus_test.txt') #essentially has the same behavior with loadTrainData but loads testing data

#TUNING PARAMETERS
tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000, 10000]}]

scores = ['precision_macro', 'recall_macro', 'accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(train_images.flatten().reshape(649, 1024), train_labels)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full training set.")
    print("The scores are computed on the full test set.")
    print()
    y_true, y_pred = test_labels, clf.predict(test_images.flatten().reshape(183,1024))
    print(classification_report(y_true, y_pred))
    print()

#Select best model
clf1 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.1).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf2 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.01).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf3 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf4 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.0001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf5 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.1).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf6 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.01).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf7 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf8 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.0001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf9 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.1).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf10 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.01).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf11 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf12 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.0001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf13 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.1).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf14 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.01).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf15 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf16 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.0001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf17 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.1).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf18 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.01).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf19 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.001).fit(train_images.flatten().reshape(649, 1024),train_labels)
clf20 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.0001).fit(train_images.flatten().reshape(649, 1024),train_labels)

print (clf1.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf2.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf3.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf4.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf5.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf6.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf7.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf8.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf9.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf10.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf11.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf12.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf13.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf14.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf15.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf16.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf17.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf18.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf19.score(test_images.flatten().reshape(183,1024),test_labels))
print (clf20.score(test_images.flatten().reshape(183,1024),test_labels))

#45.4, 24.0, 18.6, 7.1, 52.5, 51.4, 25.1, 18.0, 50.8, 53.0, 51.4, 27.3, 52.5, 51.4, 53.0, 49.7, 51.4, 51.9, 51.9, 51.4
#Best precision, recall, training accuracy for C=10000 gamma=.01 Test Accuracy 51.9%
#Best test accuracy 53% for C= 100 gamma=0.01 and C= 1000 gamma=0.001