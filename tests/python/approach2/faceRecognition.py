import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import functions as fn
import sys
import random
import glob
import cv2
import time

random.seed(42)

'''
Set variables
'''

images_per_class = 30
number_person_testing=1140
epsilon=0.01
width = 12
height = 10
threshold = 0.01


'''
Load data from Extended Yale B. Each subject has at least
60 images of his/her face. There are no images of subject 14
'''
# Store the path to the images for each subject.
images_subjects = []
for directory in glob.glob("CroppedYale/*"):
   images_subjects.append(glob.glob(directory+"/*.pgm"))

number_classes = len(images_subjects)


'''
Make a matrix with the images from all classes
'''
# Vector of matrices
A = []

for id_number in range(number_classes):
   
   mat_images_person = [] 
   
   for im in random.sample(images_subjects[id_number],k=images_per_class):
      a = cv2.imread(im,0)
      a_resized = cv2.resize(a,(width,height),interpolation = cv2.INTER_AREA)
      A.append(a_resized.flatten('F'))
      images_subjects[id_number].remove(im)

A = np.asmatrix(A).T
A_norm = normalize(A, axis=0, norm='l2')   
   
print(" ")
print("Got matrix that contains info for all classes")
print(" ")


'''
Start making the algorithm.
'''

def testing_accuracy():
   print("**************************************")
   print("      Start checking accuracy...")
   print(" ")
   i=0
   counter = 0
   testing_each_class=int(number_person_testing/number_classes)
   
   for id_number in range(number_classes):
      testing_images = random.sample(images_subjects[id_number],k=testing_each_class)
      
      for image in testing_images:
         
         class_image = fn.classify(image,width,height,number_classes,images_per_class,A_norm,epsilon,threshold)
         
         if class_image == -1 :
            print("Image is not a person in the dataset")
         else:
            if class_image == id_number:
               #print("Image was correctly classified")
               i = i+1
      
         counter = counter + 1
         
         progress = counter*100/number_person_testing
         
         if progress % 10 == 0:
            print("Overall progress ",progress,"%")
   
   print(" ")
   print("Percentage of accuracy:", i*100/number_person_testing,"%")

start_time = time.time()
testing_accuracy()
end_time = time.time()

print(" ")
print("Time it took to classify",number_person_testing,"images was",end_time-start_time,"s")

#*******************************************************************************************
# Plotting images

# img = cv2.imread(images[0][1],0)
# cv2.imshow('image',img)
# cv2.waitKey(0)