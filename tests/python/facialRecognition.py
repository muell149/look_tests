import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import scipy.misc
import functions as fn
import sys
import random

'''
Load data from Labeled Faces in the Wild. We will only use
persons with at least 100 images of their face.
'''

lfw_dataset = fetch_lfw_people(min_faces_per_person=100,color=False, resize=.16)

# Shape and number of samples
n_samples, h, w = lfw_dataset.images.shape
print(h,w)
# Vector with components as the flattened vectors with the images
images = lfw_dataset.data

# Id of each image
ordered_id = lfw_dataset.target

# Names of the person in each image
target_names = lfw_dataset.target_names

# Split into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    images, ordered_id, test_size=0.25, random_state=42)


'''
Make a matrix for each class (or person). Each row will be a 
different image of the same person.
'''
# Vector of matrices
A = []

for id_number in range(len(target_names)):
   mat_images_person = [] 
   for i in range(len(y_train)):
      if y_train[i]==id_number:
         mat_images_person.append(x_train[i])
   A.append(np.asmatrix(mat_images_person))


'''
Set the rank as in the article
'''
rank = 0.2 * h * w


'''
Start making the algorithm.
'''
M_class = []
L_class = []

print("Almost...")
sys.exit()

# Iteration over the different classes (or persons)
for matrix in A:
   
   # Get low rank approximation
   M = fn.low_rank_approx(matrix)#<-Problem with rank

   # Optimization part
   L = fn.optimization(M)

   M_class.append(M)
   L_class.append(M)

'''
Identification
'''

for index in random.sample(range(0, len(y_test)), 5):
   print("An image labeled with the index ", y_test[index], "has been identified with the index ", identify(x_test[index],vec_M,vec_L))   

   
   
   