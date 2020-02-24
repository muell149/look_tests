import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import normalize
import scipy.misc
import functions as fn
import sys
import random
import timeit

'''
Load data from Labeled Faces in the Wild. We will only use
persons with at least 100 images of their face.
'''

lfw_dataset = fetch_lfw_people(min_faces_per_person=50,color=False, resize=.3)

# Shape and number of samples
_, h, w = lfw_dataset.images.shape

# Vector with components as the flattened vectors with the images
images = lfw_dataset.data

# Id of each image
ordered_id = lfw_dataset.target

# Names of the person in each image
target_names = lfw_dataset.target_names

# Split into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    images, ordered_id, test_size=0.25, random_state=42)

number_classes=len(target_names)

'''
Set variables
'''
images_per_class = 13
number_person_testing=100
epsilon=.01

'''
Make a matrix with the images from all classes
'''
# Vector of matrices
A = []

for id_number in range(number_classes+1):
   mat_images_person = [] 
   for i in range(len(y_train)):
      if y_train[i]==id_number:
         mat_images_person.append(x_train[i])
   mat_images_person = mat_images_person[:images_per_class]
   for j in mat_images_person:
      A.append(j)

A=np.asmatrix(A).T
A_norm = normalize(A, axis=0, norm='l2')   
   
print(" ")
print("Got matrix that contains info for all classes")
print(" ")

'''
Start making the algorithm.
'''

def testing_accuracy():
   i=0
   for index in random.sample(range(0, len(y_test)), number_person_testing):
      
      Y=np.asmatrix(x_test[index]).T
      X=fn.optimization(A_norm,Y,epsilon)
      r = []
      for class_index in range(1,number_classes+1):
         X_g = fn.deltafunction(class_index,images_per_class,number_classes,X)
         r.append(np.linalg.norm(Y-A*X_g,2))
      
      # print("An image of  ", target_names[y_test[index]], 
      #       "has been identified as ", target_names[np.argmin(r)])
      
      if target_names[y_test[index]]==target_names[np.argmin(r)]:
          i = i+1
   
   print("Percentage of accuracy:", i*100/number_person_testing,"%")

testing_accuracy()