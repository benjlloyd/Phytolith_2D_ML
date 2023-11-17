from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as scl
from sklearn import svm, grid_search
from LoadData import LoadData
from LoadData import TribeLoader
from sklearn.model_selection import train_test_split #being deprecated, works for now
from sklearn.model_selection import GridSearchCV #being deprecated, works for now
from sklearn.metrics import classification_report
from sklearn.svm import SVC

np.random.seed(13)
 
tl = TribeLoader(32) # your images will be size * size
train_images_max, train_labels_max, train_tribe_max = tl.load('/Users/jessicaperry/Dropbox/phytoliths/max_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_train.txt')
test_images_max, test_labels_max, test_tribe_max = tl.load('/Users/jessicaperry/Dropbox/phytoliths/max_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_test.txt') #essentially has the same behavior with loadTrainData but loads testing data
train_images_mean, train_labels_mean, train_tribe_mean = tl.load('/Users/jessicaperry/Dropbox/phytoliths/mean_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_train.txt')
test_images_mean, test_labels_mean, test_tribe_mean = tl.load('/Users/jessicaperry/Dropbox/phytoliths/mean_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_test.txt') #essentially has the same behavior with loadTrainData but loads testing data
train_images_median, train_labels_median, train_tribe_median = tl.load('/Users/jessicaperry/Dropbox/phytoliths/median_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_train.txt')
test_images_median, test_labels_median, test_tribe_median = tl.load('/Users/jessicaperry/Dropbox/phytoliths/median_images', list_file = '/Users/jessicaperry/Dropbox/phytoliths/tribe_test.txt') #essentially has the same behavior with loadTrainData but loads testing data

args1 = (train_images_max, train_images_mean, train_images_median)
args2 = (test_images_max, test_images_mean, test_images_median)
args3 = (train_labels_max, train_labels_mean, train_labels_median)
args4 = (test_labels_max, test_labels_mean, test_labels_median)
#args5 = (train_tribe_max, train_tribe_mean, train_tribe_median, train_tribe_focus, train_tribe_maxgan, train_tribe_meangan, train_tribe_gan, train_tribe_focusgan)
#args6 = (test_tribe_max, test_tribe_mean, test_tribe_median, test_tribe_focus, test_tribe_maxgan, test_tribe_meangan, test_tribe_gan, test_tribe_focusgan)

train_images = np.concatenate(args1)
test_images = np.concatenate(args2)
train_labels = np.concatenate(args3)
test_labels = np.concatenate(args4)
#train_tribe = np.concatenate(args5)
#test_tribe = np.concatenate(args6)
print (train_images.shape)
print (test_images.shape)

#TUNING PARAMETERS
tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000, 10000]}]

scores = ['precision_macro', 'recall_macro', 'accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(train_images.flatten().reshape(2004, 1024), train_labels)

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
    y_true, y_pred = test_labels, clf.predict(test_images.flatten().reshape(507,1024))
    print(classification_report(y_true, y_pred))
    print()

#Select best model
clf1 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.1).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf2 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.01).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf3 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf4 = svm.SVC(kernel= 'rbf', C= 1, gamma=0.0001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf5 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.1).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf6 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.01).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf7 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf8 = svm.SVC(kernel= 'rbf', C= 10, gamma=0.0001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf9 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.1).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf10 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.01).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf11 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf12 = svm.SVC(kernel= 'rbf', C= 100, gamma=0.0001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf13 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.1).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf14 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.01).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf15 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf16 = svm.SVC(kernel= 'rbf', C= 1000, gamma=0.0001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf17 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.1).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf18 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.01).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf19 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.001).fit(train_images.flatten().reshape(2004, 1024),train_labels)
clf20 = svm.SVC(kernel= 'rbf', C= 10000, gamma=0.0001).fit(train_images.flatten().reshape(2004, 1024),train_labels)

print (clf1.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf2.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf3.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf4.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf5.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf6.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf7.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf8.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf9.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf10.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf11.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf12.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf13.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf14.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf15.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf16.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf17.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf18.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf19.score(test_images.flatten().reshape(507,1024),test_labels))
print (clf20.score(test_images.flatten().reshape(507,1024),test_labels))

#79.3, 71.0, 64.3, 61.9, 80.9, 78.7, 67.9, 64.3, 81.5, 75.9, 73.2, 64.5, 81.5, 77.7, 72.0, 71.4, 81.5, 77.9, 72.4, 67.5
#Best precision, training accuracy for C=10000 gamma=.01 Test Accuracy 77.9
#Best recall for C=10000 gamma=.001 test accuracy 72.4
#Best test accuracy 81.5 for C=10000 gamma=.1