from cvxopt import matrix, spmatrix, normal, setseed, blas, lapack, solvers
import nucnrm
import matplotlib.pyplot as plt

# Solves a randomly generated nuclear norm minimization problem 
#
#    minimize || A(x) + B ||_*
#
# with n variables, and matrices A(x), B of size p x q.

setseed(0)

p, q, n = 100, 100, 100
A = normal(p*q, n)
B = normal(p, q)

# options['feastol'] = 1e-6
# options['refinement'] = 3

sol = nucnrm.nrmapp(A, B)
print("hola")
x = sol['x']
Z = sol['Z']

s = matrix(0.0, (p,1))
X = matrix(A*x, (p, q)) + B
lapack.gesvd(+X, s)
nrmX = sum(s)
lapack.gesvd(+Z, s)
nrmZ = max(s)
res = matrix(0.0, (n, 1))
blas.gemv(A, Z, res, beta = 1.0, trans = 'T')

print("\nNuclear norm of A(x) + B: {}".format(nrmX))
print("Inner product of B and Z: {}".format(blas.dot(B, Z)))
print("Maximum singular value of Z: {}".format(nrmZ))
print("Euclidean norm of A'(Z): {}".format(blas.nrm2(res)))
