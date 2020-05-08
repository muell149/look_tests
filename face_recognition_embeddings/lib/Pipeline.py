import glob
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

detector = MTCNN()
embedder = FaceNet()

class DataSet:
	def __init__(self, directory, extension, size, threshold):

		self.train_images = {}
		self.test_images_known = {}
		self.test_images_unknown = {}
		self.test_images_group = []
		self.classes = {}
		self.size=size
		self.threshold= threshold

		aux = glob.glob( directory + "/Train/*" )
		for d in aux:
			images = glob.glob(d + "/*." + extension)
			self.train_images[d.split("/")[-1]]=images

		self.subjects_number = len(self.train_images)
		self.index_to_subject = {}
		self.subject_to_index = {}

		for i,subject in enumerate(self.train_images):
			self.index_to_subject[i]=subject
			self.subject_to_index[subject]=i

		aux = glob.glob( directory + "/Test/*" )
		for d in aux:
			images = glob.glob(d + "/*." + extension)
			self.test_images_known[d.split("/")[-1]]=images

		aux = glob.glob( directory + "/Unknown/*" )
		for d in aux:
			images = glob.glob(d + "/*." + extension)
			self.test_images_unknown[d.split("/")[-1]]=images

		aux = glob.glob( directory + "/Group" )
		for d in aux:
			images = glob.glob(d+ "/*." + extension)
			for i in images:
				self.test_images_group.append(i)

	def print_dataset_info(self):
		print("\n\n")
		print("*"*50,"\n*             DATASET INFORMATION                *")
		print("*"*50,"\n")
		print("Number of subjects for training:", self.subjects_number)
		print("\n")

	def load_model(self,name,train=False):
		if train:
			print("\n\n")
			print("*"*50,"\n*                START TRAINING                  *")
			print("*"*50,"\n")
			if not os.path.exists('models'):
				os.makedirs('models')

			# Getting the arrays for the training
			train_y_subjects, train_x = load_set(self.train_images, size = 160)

			# Getting embeddings from FaceNet and normalizing them
			train_embeddings = embedder.embeddings(train_x)
			train_x = Normalizer(norm='l2').transform(train_embeddings)

			# Encoder in order to associate labels with a certain number
			train_y = []
			for subject in train_y_subjects:
				train_y.append(self.subject_to_index[subject])

			train_y = np.asarray(train_y)

			model = SVC(kernel='linear', probability=True)

			model.fit(train_x, train_y)

			pickle.dump(model, open('models/{}.sav'.format(name), 'wb'))

			self.model = pickle.load(open('models/{}.sav'.format(name), 'rb'))
		else:
			self.model = pickle.load(open('models/{}.sav'.format(name), 'rb'))

	def test_model(self):
		print("\n\n")
		print("*"*50,"\n*              START TESTING (Known)               *")
		print("*"*50,"\n")
		test_y_subjects, test_x = load_set(self.test_images_known, size = self.size)

		test_embeddings = embedder.embeddings(test_x)
		test_x = Normalizer(norm='l2').transform(test_embeddings)

		# Encoder in order to associate labels with a certain number
		test_y = []
		for subject in test_y_subjects:
			test_y.append(self.subject_to_index[subject])

		test_y = np.asarray(test_y)

		y_test_pred = self.model.predict(test_x)
		y_test_proba = self.model.predict_proba(test_x)
		y_aux=[]
		for pred, proba in zip(y_test_pred,y_test_proba):
			y_aux.append(identify_unknown(proba,pred,self.threshold))

		score_test_known = accuracy_score(test_y,y_aux)
		print("Accuracy on test:",score_test_known*100,"%\n\n")
		return score_test_known*100

def identify_unknown(x,index,t):
	np.delete(x,index)
	for i in x:
		if i<=t:
			return index
		else:
			return -1

def load_set(set,size):
	images = []
	labels = []
	for person in set:
		for path in set[person]:
			labels.append(person)
			images.append(extract_face(path,size))
		print("Got",len(set[person]),"for subject",person)
	print("\n")
	return np.asarray(labels),np.asarray(images)


def extract_face(filename,size):

	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	results = detector.detect_faces(image)

	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = image[y1:y2, x1:x2]

	face_array = cv2.resize(face,(size,size),interpolation=cv2.INTER_NEAREST)

	return face_array


'''
def load_dataset(directory, size=160):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path, size=size)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)


import cv2
import logging
import numpy as np
import os
import time
from PIL import Image
# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import glob



# extract a single face from a given photograph



# load images and extract faces for all images in a directory
def load_faces(directory, size=160):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path,required_size=(size, size))
		# store
		faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, size=160):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path, size=size)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
'''