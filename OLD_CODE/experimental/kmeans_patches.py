from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as scl
from sklearn import svm, grid_search
from LoadData import LoadData
from LoadData import TribeLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction import image
from sklearn.svm import SVC

np.random.seed(13)

ld = LoadData(32) # your images will be size * size
train_images, train_labels = ld.loadTrainData('/Users/jessicaperry/Dropbox', imageStyle = 'max_images')
    #"images" are (num_of_images, size, size) shaped np array, "labels" are (num_of_images,) shaped np array
test_images, test_labels = ld.loadTestData('/Users/jessicaperry/Dropbox', imageStyle = 'max_images') #essentially has the same behavior with loadTrainData but loads testing data

#Select random m 16x16 patches
m = 100000
patch_error = 10
#error_zca = .01 #for 16x16 and .1 for 8x8
error_zca = .01*np.identity(256)
k = 100 #Number of centroids
index1 = np.random.randint(0, 777, m)
index2 = np.random.randint(0, 289, m)
patches_all = np.zeros((777, 289, 16, 16))
patches = np.zeros((m, 16, 16))

#Convert all images to patches
for i in range(0, len(train_images)):
	patches_all[i] = image.extract_patches_2d(train_images[i], (16, 16))

#Select a random 100,000 from all patches
for i in range(0, m):
	patches[i] = patches_all[index1[i], index2[i],:,:]

patches2 = patches.reshape(m, 256)

#Normalize inputs, x(i) = (x(i) - mean x(i)) / sqrt(var(x(i)) + error_norm)
for i in range(0, m):
	#patches[i,:,:] = (patches[i,:,:] - np.mean(patches[i,:,:]))/(np.sqrt(np.var(patches[i,:,:]) + patch_error))
	patches2[i,:] = (patches2[i,:] - np.mean(patches2[i,:]))/(np.sqrt(np.var(patches2[i,:]) + patch_error))

	
#N = patches.shape[2]
#r = patches[i,:,:]
#r -= np.sum(r, axis=1, keepdims=True) / N
#cov = np.dot(r, r.T)  /(N - 1)

#Whiten inputs
V, D = np.linalg.eig(np.cov(patches2.T)) #so VDV.T = cov(x)
print (np.amin(D))
for i in range(0, m):
	patches2[i] = np.dot(np.dot(V, np.dot(np.diag(1.0/np.sqrt(D + error_zca)), V.T)),patches2[i]) #x(i) = V(D + error_zca * I)^(-1/2) V.T x(i)

#Initialize K-Means
c = np.random.randn(k) #Initialize centroids from normal dist. and normalize to unit length
c = c/np.linalg.norm(c)

#Damped Centroids
#def damped(D, X, S)
	#D = 
	#At each iteration compute new centroids
	#D_new = argmin_D ||DS - X||^2_2 + ||D-D_old||^2_2
		#= (SS.T + I)^(-1) (XS.T + D_old)
	#D_new = normalize(D_new)

#K-Means
#Loop until convergence (usually 10 is enough)
	#s(i)_j = D(j).T x(i) if j == argmax_l |D(l).T x(i)| for all j,i
	#		= 0 			else
	#D = XS.T + D							#Damped centroids
	#D(j) = D(j) / ||D(j)||_2 for all j		#Damped centroids
	
#Constraints
	#||s(i)||_0 <= 1 for all i   only one non zero entry, not for Sparse Coding
	#||D(j)||_2 = 1 for all j	 each column is unit length 1


#Spherical K-Means
	#min_D,s sum of i ||Ds(i) - x(i)||^2_2

#Sparse Coding
	#min_D,s sum of i ||Ds(i) - x(i)||^2_2 + lambda||s(i)||_1

#Notes on dimensionality
# D(j)	- n x 1
# D 	- n x k
# x(i) 	- n x 1 
# X 	- n x m
# s(i) 	- k x 1
# S		- k x m

# i		1..m
# j		1..k