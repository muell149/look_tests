import numpy as np
import cvxpy as cp
import sys

def optimization(M, lamb):
   '''
   Solves the following optimization problem
   
       min      || Z ||_*  +  lambda || M - MZ ||_{F}^2
      Z,L,E
      
      
      subject to: 
      
      M = MZ + LM + E
      
      where
      
      M =         U_{M} S_{M} V^T_{M} is the Skinny SVD of M
      Z =         V_{M} W_{Z} V^T_{M}
      L =         U_{M} ( I - W_{Z} ) U_{M}^T
      rank(Z) =   r
   
      ||.||_*   is called the nuclear norm
      ||.||_F   is called the frobenius norm
   '''
   
   # First, get the skinny SVD of M in order to obtain U and V
   print("Starting optimization")
   print("  Getting Skinny SVD...")
   U, _, V = skinny_SVD(M) 
   print("  Got Skinny SVD")

   #***************** Problem construction ****************
   
   # Define variables
   mr, mc = M.shape
   _ , _ = U.shape
   _, vc = V.shape
   
   W = cp.Variable((vc,vc)) 
   E = cp.Variable((mr,mc)) 
   
   # Objective function definition
   print("  Defining objective function")
   objective = cp.Minimize( cp.norm(V*W*V.T, "nuc") 
                           + lamb * (cp.norm(M-M*V*W*V.T, 'fro')) **2.0 )
   
   # Constraint definition
   print("  Define constraint")
   constraints = [ M == M*V*W*V.T + U*(np.identity(vc) - W)*U.T*M + E] # <- Rank constraint is missing

   # Solve problem using cvxpy
   
   print("  Define problem to solve")
   prob = cp.Problem(objective,constraints)

   prob.solve(verbose=True)
   print("="*30,"Problem solved","="*30)
   print("="*76)
   
   L = U*(np.identity(W.value.shape[0])-W.value)*U.T
   
   return L

def identify(y,vec_M,vec_L):
   '''
   kNN definition of distance to a class
   
   Input
   
   y:       Flattened image to be classified
   vec_M:   Vector of the M's matrices that define a class
   vec_L:   Vector of the L's matrices that define a class
   
   Output
   
   An index that defines the class to which the image y belongs
   '''
   
   e = np.empty(len(vec_M))
   
   for i in range(len(vec_M)):
      e[i] = (np.linalg.norm(vec_L[i].getH()*np.asmatrix(y).T-vec_L[i].getH()*vec_M[i],2))**2
   
   return np.argmin(e)

def low_rank_approx(A, r=1):
   '''
   Computes an r-rank approximation of a matrix A (m x n)
   by SVD. 

   The decomposition is A (m x n) = U (m x m) S (m x n) V (n x n)

   The columns of U and columns of V are the left and right singular
   vectors, respectively.
   '''

   u, s, v = np.linalg.svd(A, full_matrices=False)

   D = np.diag(s)

   D1 = D.copy()

   D1[D1 < s[int(r)]] = 0.

   M = u.dot(D1).dot(v)

   return M

def skinny_SVD(A):
   '''
   For a matrix A of rank r, its skinny SVD is computed by 
   A = U_r S_r V^T_r, where S_r = diag(s_1,s_2,...,s_r) with
   s_i being positive singular values
   '''
   # Get SVD of A
   u_aux, s_aux, v_aux = np.linalg.svd(A,full_matrices=False)

   # Get rank r of A
   r = np.linalg.matrix_rank(A)

   # Take only the first r singular values (the singular values
   # are always positive), and the first r columns of u and 
   # the first r rows of v transpose
   s = s_aux[:r]
   u = u_aux[:,0:r]
   v = v_aux[0:r,:]
   
   return u,s,v.T