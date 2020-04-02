import cv2
import glob
import random
import numpy as np
import cvxpy as cp

from lib.Pipeline import detect_and_align
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

class DataSet:
    def __init__(self, dir, ext, images_per_class, height, width, epsilon, threshold):
        # images_subjects = []

        self.test_images_known = []
        self.test_images_unknown = []

        self.height = height
        self.width = width
        self.images_per_class = images_per_class

        self.epsilon = epsilon
        self.threshold = threshold

        A = []
        for directory in glob.glob(dir):
            images = glob.glob(directory + "/*." + ext)

            if len(images) > images_per_class:
                subject_images_sample = random.sample(images, k=images_per_class+1)
                self.test_images_known.append(subject_images_sample.pop())

                aligned_images = [detect_and_align(im, width, height) for im in subject_images_sample]
                if any(a is None for a in aligned_images):
                    continue
                else:
                    print("{:<50} | {:<5}".format(directory.split("/")[-1], len(images)))
                    for a in aligned_images:
                        A.append(a.flatten("F"))
            else:
                self.test_images_unknown.append(random.choice(images))

        self.matrix = normalize(np.asmatrix(A).T, axis=0, norm="l2")
        self.number_classes = int(self.matrix.shape[1]/self.images_per_class)


    def classify(self, image, plot=True):
        aligned = detect_and_align(image, self.width, self.height)
        if aligned is None:
            print("No face detected")
            return None

        Y = np.asmatrix(aligned.flatten("F")).T

        X, e = optimization(self.matrix, Y, self.epsilon)

        delta_l = []
        for class_index in range(1, self.number_classes+1):
            X_g = deltafunction(class_index, self.images_per_class, self.number_classes, X)
            delta_l.append(X_g)

        e_r = []
        for class_index in range(0, self.number_classes):
            e_r.append(np.linalg.norm(Y-e-self.matrix*delta_l[class_index], 2))

        if plot==True:
            plt.plot(e_r,'o')
            plt.xlabel("Subject")
            plt.ylabel(r"Error $||y-A\delta_i||_2$")
            plt.grid()
            plt.show()

        if sci(X, delta_l) >= self.threshold:
            return np.argmin(e_r)

        else:
            #print("Image is not a person in the dataset")
            return -1


def optimization(A, Y, epsilon, print_proc=False):
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
