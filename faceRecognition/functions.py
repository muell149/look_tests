import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
import cv2

def optimization(A,Y,epsilon,max_iters):
   '''
   Solves the following optimization problem
   
       min      || X ||_1
        X
      
      
      subject to: 
      
      AX-Y <= epsilon
      
      where
      
      A = Matrix that contains info for all classes (subjects)
      X = Sparse coeficient vector, we look for it to be zero in
          most of its components, except for the ones
          representing one subject
      Y = Image to recognize
   
      ||.||_1   is called the l_1 norm
   '''
   x_rows = A.shape[1]
   # The variable, has to has the same rows as the matrix A
   X = cp.Variable((x_rows,1)) 

   # Objective function definition
   objective = cp.Minimize(cp.norm(X, 1))
   
   # Constraint definition
   constraints = [cp.norm(A*X-Y,2)<=epsilon] 
   
   # Solve problem using cvxpy
   prob = cp.Problem(objective,constraints)
   #prob.solve(solver=cp.SCS,gpu=False,use_indirect=True,max_iters=max_iters,verbose=False)
   prob.solve()

   return X.value

def deltafunction(class_index,images_per_class,number_classes,X):
   '''
   Funtion that returns a vector of the same size as X 
   whose only nonzero entries are the ones in X that are associated
   with "class_index"
   '''
   
   m=images_per_class*number_classes
   
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

def classify(image,width,height,number_classes,images_per_class,A,epsilon,threshold,max_iters,plot):
   '''
   Function to classify image.
   '''
   
   a = cv2.imread(image,0)
   
   # Original image is resized 
   a_resized = cv2.resize(a,(width,height),interpolation=cv2.INTER_AREA)
   
   # Resize image is flattened
   Y = np.asmatrix(a_resized.flatten('F')).T
   
   # Solve the optimization problem
   X = optimization(A,Y,epsilon,max_iters)
   
   delta_l = []
   
   for class_index in range(1,number_classes+1):
      X_g = deltafunction(class_index,images_per_class,number_classes,X)
      delta_l.append(X_g)
   
   e_r = []
      
   for class_index in range(0,number_classes):
      e_r.append(np.linalg.norm(Y-A*delta_l[class_index],2))
   
   if plot==True:
      plt.plot(e_r,'o')
      plt.xlabel("Subject")
      plt.ylabel(r"Error $||y-A\delta_i||_2$")
      plt.grid()
      plt.show()
   
   if sci(X,delta_l) >= threshold:      
      return np.argmin(e_r)
            
   else:
      return -1
