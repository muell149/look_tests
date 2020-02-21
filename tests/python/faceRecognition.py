import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import scipy.misc
import functions as fn
import sys
import random
import timeit

'''
Load data from Labeled Faces in the Wild. We will only use
persons with at least 100 images of their face.
'''

lfw_dataset = fetch_lfw_people(min_faces_per_person=50,color=False, resize=.16)

# Shape and number of samples
n_samples, h, w = lfw_dataset.images.shape

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
Set variables
'''
rho = 0.02
rank = rho * h * w
number_images_per_person = 30
lamb = .05
number_person_testing=250

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
         mat_images_person.append(x_train[i]/255)
   mat_images_person = mat_images_person[:number_images_per_person]
   A.append(np.asmatrix(mat_images_person).T)
   
print(" ")
print("Got matrix that contains info from each class")
print(" ")

'''
Start making the algorithm.
'''
M_class = []
L_class = []

# Iteration over the different classes (or persons)
for index in range(len(A)-1):
   t = len(target_names[index])
   print(" ")
   print(" ")
   print(" ")
   print("*"*(int((47-t)/2))," Start Algorithm for",target_names[index],"Class ","*"*(int((47-t)/2)))
   print(" ")
   print("Class ",index+1,"/",len(A),sep='')
   print(" ")
   print("Getting clean information matrix...")

   # Get low rank approximation
   M = fn.low_rank_approx(A[index],rank)

   print("Got clean information")
   
   # Optimization part
   L = fn.optimization(M,lamb)
   
   
   M_class.append(M)
   L_class.append(M)

print(" ")
print("Vectors M and L obtained")
print(" ")
print(" ")
'''
Identification
'''

print("Starting identification")
def testing_accuracy():
   
   i=0
   for index in random.sample(range(0, len(y_test)), number_person_testing):
      # print("An image of  ", target_names[y_test[index]], 
      #       "has been identified as ", target_names[fn.identify(x_test[index],M_class,L_class)])
      if target_names[y_test[index]]==target_names[fn.identify(x_test[index],M_class,L_class)]:
         i = i+1
         
   print("Percentage of accuracy:", i*100/number_person_testing,"%")
   
elapsed_time = timeit.timeit(testing_accuracy,number=1)
print("Elapsed time to identify",number_person_testing, "people is: ", elapsed_time,"s")