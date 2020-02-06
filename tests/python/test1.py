import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

#********************* Test functions ************************
def plotimagewomean(x):
   b = x.mean(0)
   
   plt.figure(1)
   plt.imshow(x-b,cmap=plt.cm.gray)

   plt.figure(2)
   plt.imshow(x,cmap=plt.cm.gray)

   plt.show()
#*************************************************************


# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=100,color=False)

n_samples, h, w = lfw_dataset.images.shape

X = np.reshape(lfw_dataset.images,(h*w,n_samples))
U, Sigma, D = np.linalg.svd(X, full_matrices=True)

print(X.shape)
print(U.shape)
print(U.shape)
print(D.shape)

a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
u, s, vh = np.linalg.svd(a, full_matrices=True)
print(a.shape)
print(u.shape)
print(s.shape)
print(vh.shape)