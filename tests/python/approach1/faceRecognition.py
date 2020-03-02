import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import functions as fn
import sys
import random
import glob
import cv2
import time
import json

random.seed(42)

'''
Set variables
'''
saveML = True
images_per_class = 30
number_person_testing=1140
width = 12
height = 10
rho = 0.02
rank = rho * height * width
lamb = .05


'''
Load data from Extended Yale B. Each subject has at least
60 images of his/her face. There are no images of subject 14
'''
# Store the path to the images for each subject.
images_subjects = []
for directory in glob.glob("../approach2/CroppedYale/*"):
   images_subjects.append(glob.glob(directory+"/*.pgm"))

number_classes = len(images_subjects)


if saveML == False:
   '''
   Make a matrix for each class (or person). Each row will be a 
   different image of the same person.
   '''
   # Vector of matrices
   A = []

   for id_number in range(number_classes):
      
      mat_images_person = [] 
      for im in random.sample(images_subjects[id_number],k=images_per_class):
         a = cv2.imread(im,0)
         a_resized = cv2.resize(a,(width,height),interpolation = cv2.INTER_AREA)
         mat_images_person.append(a_resized.flatten('F'))
         images_subjects[id_number].remove(im)
      
      A.append(normalize(np.asmatrix(mat_images_person).T,axis=0,norm='l2'))

   print(" ")
   print(" Got matrix that contains info for", number_classes,"subjects,\n using",images_per_class,"images of each subject.")
   print(" ")

   '''
   Start making the algorithm.
   '''
   M_class = []
   L_class = []

   # Iteration over the different classes (or persons)
   print("Start Algorithm for each class")

   for index in range(number_classes):
      
      print("Class ",index+1,"/",number_classes,sep='')
      print(" ")
      print("Getting clean information matrix...")

      # Get low rank approximation
      M = fn.low_rank_approx(A[index],rank)

      print("Got clean information")
      
      # Optimization part
      L = fn.optimization(M,lamb)
      
      
      M_class.append(M)
      L_class.append(L)

   np.save('M_class.npy', M_class)
   np.save('L_class.npy', L_class)
      

M_class = np.load('M_class.npy')
L_class = np.load('L_class.npy')

print(" ")
print("Vectors M and L obtained")
print(" ")
print(" ")

'''
Identification
'''
print("Starting identification")
print(" ")
i=0
counter = 0
testing_each_class=int(number_person_testing/number_classes)


for id_number in range(number_classes):
   testing_images = random.sample(images_subjects[id_number],k=testing_each_class)

   for image in testing_images:
      
      a = cv2.imread(image,0)
   
      # Original image is resized 
      a_resized = cv2.resize(a,(width,height),interpolation=cv2.INTER_AREA)
      
      # Resize image is flattened
      Y = np.asmatrix(a_resized.flatten('F')).T
      
      identity = fn.identify(Y,M_class,L_class)
      
      if identity == id_number:
         i = i + 1
         
      # else:
      #    print(identity,id_number)
      #    print("Incorrect identification")
   
      counter = counter + 1
      
      progress = counter*100/number_person_testing
      
      if progress % 10 == 0:
         print("Overall progress ",progress,"%")
   
print(" ")
print("Percentage of accuracy:", i*100/number_person_testing,"%")