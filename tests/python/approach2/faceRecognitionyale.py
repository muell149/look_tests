import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import normalize
import scipy.misc
import functions as fn
import sys
import random
import glob
import os
import timeit
import cv2
import time

'''
Load data from Extended Yale B. Each subject has at least
60 images of his/her face. There are no images of subject 14
'''
# Store the path to the images for each subject.
images_subjects = []
for directory in glob.glob("CroppedYale/*"):
   images_subjects.append(glob.glob(directory+"/*.pgm"))

number_classes = len(images_subjects)

# Plotting images
# img = cv2.imread(images[0][1],0)
# cv2.imshow('image',img)
# cv2.waitKey(0)

'''
Set variables
'''
images_per_class = 30
number_person_testing=300
epsilon=0.05
width = 12
height = 10

'''
Make a matrix with the images from all classes
'''
# Vector of matrices
A = []

for id_number in range(number_classes):
   
   mat_images_person = [] 
   
   for im in images_subjects[id_number][:images_per_class]:
      a = cv2.imread(im,0)
      a_resized = cv2.resize(a,(width,height),interpolation = cv2.INTER_AREA)
      A.append(a_resized.flatten('F'))

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
   testing_each_class=int(number_person_testing/number_classes)
   
   for id_number in range(number_classes):
      testing_images = random.choices(images_subjects[id_number][images_per_class:],k=testing_each_class)
      
      for image in testing_images:
         
         a = cv2.imread(image,0)
         a_resized = cv2.resize(a,(width,height),interpolation = cv2.INTER_AREA)

         Y=np.asmatrix(a_resized.flatten('F')).T
         X=fn.optimization(A_norm,Y,epsilon)
         e_r = []
         
         for class_index in range(1,number_classes+1):
            X_g = fn.deltafunction(class_index,images_per_class,number_classes,X)
            e_r.append(np.linalg.norm(Y-A_norm*X_g,2))
            
         if np.argmin(e_r)==id_number:
            # print(np.argmin(e_r))
            # print(id_number)
            # print("Correct detection")
            i = i+1
         # else:
         #    print(np.argmin(e_r))
         #    print(id_number)
         #    print("INCORRECT")     
   
   print("Percentage of accuracy:", i*100/number_person_testing,"%")

start_time = time.time()
testing_accuracy()
end_time = time.time()

print("Time it took to classify",number_person_testing,"images was",end_time-start_time,"s")