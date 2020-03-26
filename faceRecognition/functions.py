import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
import cv2
from sklearn.preprocessing import normalize
from sklearn import random_projection
import glob
import time
import random

def getmatrix(dir,images_per_class,height,width):

   # Store the path to the images for each subject.
   images_subjects = []
   for directory in glob.glob(dir):
      images_subjects.append(glob.glob(directory+"/*.pgm"))

   images_subjects = [[p for p in pics if "Ambient" not in p] for pics in images_subjects] 

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

   A_matrix = normalize(np.asmatrix(A).T, axis=0, norm='l2')

   print("\n Got matrix that contains info for", number_classes,"subjects,\n using",images_per_class,"images of each subject.\n")
   print(" Size of each image is",height,"x",width,"\n")

   return A_matrix, number_classes, images_subjects


def optimization(A,Y,epsilon,print_proc=False):
   '''
   Solves the following optimization problem

       min      || w ||_1
        w


      subject to:

      Bw-Y <= epsilon

      where

      A = Matrix that contains info for all classes (subjects)
      X = Sparse coeficient vector, we look for it to be zero in
          most of its components, except for the ones
          representing one subject
      w = Concatenation of X(nx1) with an error vector e(mx1)
      B = Concatenation of matrix A(mxn) with the identity I(mxm)
      Y = Image to recognize

      ||.||_1   is called the l_1 norm
   '''
   I = np.asmatrix(np.identity(A.shape[0]))

   B = np.concatenate((A,I),axis=1)

   x_rows = A.shape[1]

   # The variable, has to has the same rows as the matrix A
   w = cp.Variable((x_rows+A.shape[0],1))

   # Objective function definition
   objective = cp.Minimize(cp.norm(w, 1))

   # Constraint definition
   constraints = [cp.norm(B*w-Y,2)<=epsilon]

   # Solve problem using cvxpy
   prob = cp.Problem(objective,constraints)
   prob.solve(verbose=print_proc)

   result = np.split(w.value, [x_rows, x_rows+A.shape[0]])

   X = np.asmatrix(result[0])
   e = np.asmatrix(result[1])

   return X, e

def deltafunction(class_index,images_per_class,number_classes,X):
   '''
   Funtion that returns a vector of the same size as X
   whose only nonzero entries are the ones in X that are associated
   with "class_index"
   '''

   m = images_per_class * number_classes

   d = np.asmatrix(np.zeros((m,1)))

   X = np.asmatrix(X)

   for j in range((class_index-1)*images_per_class, class_index*images_per_class):
      d[j,0]=X[j,0]

   return d

def sci(x,delta_l):
   '''
   Sparsity Concentration Index (SCI)

   It measures how concentrated the coeficients are on a
   single class on the data set. It allows us to decide
   if the vector x is represented by the images of one subject
   on the dataset
   '''

   norm_delta=[]

   for i in delta_l:
      norm_delta.append(np.linalg.norm(i,1))

   k = len(delta_l)

   return (k*max(norm_delta)/np.linalg.norm(x,1) - 1)/(k-1)

def classify(image,width,height,number_classes,images_per_class,A,epsilon,threshold,plot):
   '''
   Function to classify image.
   '''

   a = cv2.imread(image,0)

   # Original image is resized
   a_resized = cv2.resize(a,(width,height),interpolation=cv2.INTER_AREA)

   # Resize image is flattened
   Y = np.asmatrix(a_resized.flatten('F')).T

   # Solve the optimization problem
   X, e = optimization(A,Y,epsilon)

   delta_l = []

   for class_index in range(1,number_classes+1):
      X_g = deltafunction(class_index,images_per_class,number_classes,X)
      delta_l.append(X_g)

   e_r = []

   for class_index in range(0,number_classes):
      e_r.append(np.linalg.norm(Y-e-A*delta_l[class_index],2))

   if plot==True:
      plt.plot(e_r,'o')
      plt.xlabel("Subject")
      plt.ylabel(r"Error $||y-A\delta_i||_2$")
      plt.grid()
      plt.show()

   if sci(X,delta_l) >= threshold:
      return np.argmin(e_r)

   else:
      #print("Image is not a person in the dataset")
      return -1
