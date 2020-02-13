import numpy as np
import cvxopt as cx

def optimization(M):
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
   
   U_aux, S_aux, V_aux = skinny_SVD(M) 
   
   U = cx.matrix(U_aux)
   S = cx.matrix(S_aux)
   V = cx.matrix(V_aux)
   
   
   return U




















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

   return u,s,v
