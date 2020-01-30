# import numpy as np
# import cv2
# from matplotlib import pyplot as plt 

# img = cv2.imread('imaget.pgm')

import rpca
from rpca import loss

# Load "Sleep in Mammals" database
X = rpca.data.load_sleep()

# Transform it using Robust PCA
huber_loss = rpca.loss.HuberLoss(delta=1)
rpca_transformer = rpca.MRobustPCA(2, huber_loss)
X_rpca = rpca_transformer.fit_transform(X)