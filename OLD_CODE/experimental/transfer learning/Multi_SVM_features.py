import numpy as np
from sklearn import svm, grid_search
from LoadData import TribeLoader, GenusLoader, LoadData
from sklearn.model_selection import GridSearchCV  # being deprecated, works for now
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

maxScore = 0
maxCmatScore = 0
maxTuple = None
for j in np.arange(25, 30):
    train = np.load('concatenated' + '_train' + str(j) + '.npy').squeeze()
    print train.shape
    test = np.load('concatenated' + '_test' + str(j) + '.npy').squeeze()
    loader = LoadData(32)  # your images will be size * size
    style = 'max_global_autocontrast_normalized'
    _, train_labels = loader.loadTrainData('../../data', style)
    _, test_labels = loader.loadTestData('../../data', style)
    train_labels = np.concatenate((train_labels,) * 4)
    # TUNING PARAMETERS
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [1, 10, 100, 1000, 10000]}]

    for kernel in ['rbf', 'linear']:
        for gamma in [1e-2, 1e-3, 1e-4, 1e-5]:
            for C in [1, 10, 100, 1000, 10000]:
                clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                clf.fit(train, train_labels)
                score = clf.score(test, test_labels)
                cmat = confusion_matrix(test_labels, clf.predict(test))
                cmat = np.asarray(cmat.diagonal() / np.asarray(cmat.sum(axis=1), dtype=np.float32))
                cmat = np.mean(cmat)
                print score,
                if score > maxScore:
                    maxScore = score
                    maxCmatScore = cmat
                    maxTuple = (j, kernel, gamma, C, score)
                elif score == maxScore:
                    maxCmatScore = max((maxCmatScore, cmat))

            print
    print 'Max so far:', maxTuple + (maxCmatScore,)
