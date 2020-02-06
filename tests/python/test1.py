import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
 
 
# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=100,color=False)

n_samples, h, w = lfw_dataset.images.shape

# plt.imshow(lfw_dataset.images[1000].reshape(h,w),cmap=plt.cm.gray)
# plt.show()



print(lfw_dataset.images[1].reshape(h*w))