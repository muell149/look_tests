import numpy as np
import cvxpy as cp
import sys

def optimization(A,Y,epsilon):

   #***************** Problem construction ****************
   X = cp.Variable((A.shape[1],1)) 

   # Objective function definition
   objective = cp.Minimize(cp.norm(X, 1))
   
   # Constraint definition
   constraints = [cp.norm(A*X-Y,2)<=epsilon] 
   
   
   # Solve problem using cvxpy
   prob = cp.Problem(objective,constraints)
   prob.solve()
   # prob.solve(verbose=True)
   return X.value

def deltafunction(class_index,images_per_class,number_classes,X):
   m=images_per_class*number_classes
   d = np.asmatrix(np.zeros((m,1)))
   X = np.asmatrix(X)
   for j in range((class_index-1)*images_per_class, class_index*images_per_class):
      d[j,0]=X[j,0]
      
   return d