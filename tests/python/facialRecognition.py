import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
import scipy.misc
import functions as fn

'''
Load data from Labeled Faces in the Wild. We will only use
persons with at least 100 images of their face.

'''

lfw_dataset = fetch_lfw_people(min_faces_per_person=100,color=False)

n_samples, h, w = lfw_dataset.images.shape

A = np.matrix([[2,3,1],[6,9,3],[10,15,3]])

print(fn.optimization(A))

# print(np.linalg.norm(A,'nuc'))

#********************************************************************

# A = lfw_dataset.images[0]

# M = fn.low_rank_approx(A, r=2)

# plt.figure(1)
# f, ax = plt.subplots(1,2)

# ax[0].imshow(A)
# ax[1].imshow(M)
# plt.show()

#********************************************************************

# set_person_images = []

# for i in range(len(lfw_dataset.target)):
#    if lfw_dataset.target[i] == 3:
#       set_person_images.append(lfw_dataset.images[i].reshape(1,h*w)[0])
      
# A = np.matrix(set_person_images)

#*************************************************************
# Plotting an image that's inside the matrix

# plt.imshow(A[2][:].reshape(h,w))
# plt.show()

#********************************************************************