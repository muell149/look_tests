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

images_per_class        = 30
number_person_testing   = 1140
epsilon                 = 0.01
width                   = 12
height                  = 10
threshold               = 0.01
max_iters               = 200
errors_print            = False
occlude                 = False
directory               = "CroppedYale/*"


def getmatrix(dir):
   '''
   Load data from Extended Yale B. Each subject has at least
   60 images of his/her face. There are no images of subject 14
   '''
   # Store the path to the images for each subject.
   images_subjects = []
   for directory in glob.glob(dir):
      images_subjects.append(glob.glob(directory+"/*.pgm"))

   number_classes = len(images_subjects)

   '''
   Make a matrix with the images from all classes
   '''
   # Vector of matrices
   A = []

   for id_number in range(number_classes):

      for im in random.sample(images_subjects[id_number],k=images_per_class):
         a = cv2.imread(im,0)
         a_resized = cv2.resize(a,(width,height),interpolation = cv2.INTER_AREA)
         A.append(a_resized.flatten('F'))
         images_subjects[id_number].remove(im)

   A = np.asmatrix(A).T
   A_norm = normalize(A, axis=0, norm='l2')   
      
   print(" ")
   print(" Got matrix that contains info for", number_classes,"subjects,\n using",images_per_class,"images of each subject.")
   print(" ")
   print(" Size of each image is",width,"x",height)
   print(" \n")

   return A_norm, number_classes, images_subjects

def testing_accuracy(occlude,errors_print,width,height):
   '''
   Start making the algorithm.
   '''

   A_norm, number_classes, images_subjects = getmatrix(directory)

   print("********************************************")
   print("         Start checking accuracy...")
   print(" ")
   i=0
   counter = 0
   testing_each_class=int(number_person_testing/number_classes)
   errors=[]
   for id_number in range(number_classes):
      testing_images = random.sample(images_subjects[id_number],k=testing_each_class)
      
      for image in testing_images:
         try:
            class_image = fn.classify(image,width,height,number_classes,images_per_class,A_norm,epsilon,threshold,max_iters,False)
         
            if class_image == -1 :
               #print("Image is not a person in the dataset")
               pass
            else:
               if class_image == id_number:
                  #print("Image was correctly classified")
                  i = i+1
         except:
            errors.append(image)
      
         counter = counter + 1
         
         progress = counter*100/number_person_testing

         if progress % 10 == 0:
            print("Overall progress ",progress,"%")
   
   if len(errors) != 0:
      print(" ")
      print("-"*44)
      print("There were unknown errors in",len(errors),"files")
      if errors_print==True:
         print("FILES WITH ERRORS")
         for i in errors:
            print(i)
      print("-"*44)
   
   print(" ")
   print("Percentage of accuracy:", i*100/(number_person_testing-len(errors)),"%")
   return float(i*100/(number_person_testing-len(errors)))

def accuracy(width,height):
   start_time = time.time()
   accuracy_res=testing_accuracy(occlude,errors_print,width,height)
   end_time = time.time()

   print(" ")
   print("Time it took to classify",number_person_testing,"images was",end_time-start_time,"s")
   print("Average time to classify one person was",(end_time-start_time)/number_person_testing,"s")
   return [float((end_time-start_time)/number_person_testing),accuracy_res]
   
def testImage(img):

   A_norm, number_classes, _ = getmatrix(directory)

   class_image = fn.classify(img,width,height,number_classes,images_per_class,A_norm,epsilon,threshold,max_iters,True)
         
   if class_image == -1 :
      print("Image is not a person in the dataset")
   else:
      print("Image was classified as the subject", class_image)

#--------------------------------------------------------------------------------------
'''
SIMPLE TESTS
'''
#test_not_in = images_subjects.pop(10)
#test_in = images_subjects[3]

#testImage(test_not_in[5])

'''
STUDY ACCURACY AND TIME VS IMAGE SIZE
'''
#sizes = [(8,7),(9,6),(12,10),(16,14),(20,15),(24,21),(32,28),(64,56)]

#f = open("differentSizes.txt", "a")
#for (width,height) in sizes:
#   [running_time, accuracy_res] = accuracy(width,height)
#   f.write("{},{},{},{}\n".format(width,height,running_time,accuracy_res))
#f.close()