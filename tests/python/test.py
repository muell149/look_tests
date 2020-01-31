import numpy as np
import time
import cv2

# Simple matrix
#A = np.array([[3.,2.,2.],[2.,3.,-2.],[-5.,0.,1.],[4.,6.,-8.]])

# Image
A = cv2.imread("amber.jpg",0)
B = A.astype(float)


start_time = time.time()
w, u, v = cv2.SVDecomp(B)
final_time = time.time()
#print(A)
#print(w)
#print(u)
#print(v)


print((final_time-start_time)*1e6, " microseconds")