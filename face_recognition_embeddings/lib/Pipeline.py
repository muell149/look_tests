import glob
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.utils import shuffle

import sys
from joblib import dump, load
import json

import annoy

detector = MTCNN()
embedder = FaceNet()

class DataSet:
	def __init__(self, directory, extension, size, lof_nn, Ntrees):

		self.train_images = {}
		self.test_images_known = {}
		self.unknown_images = {}
		self.test_images_group = []
		self.classes = {}
		self.size=size
		self.lof_nn = lof_nn
		self.Ntrees = Ntrees

		aux = glob.glob( directory + "/Train/*" )
		for d in aux:
			d=d.replace("\\","/")
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
			d=d.replace("\\","/")
			images = glob.glob(d + "/*." + extension)
			self.test_images_known[d.split("/")[-1]]=images

		aux = glob.glob( directory + "/Unknown/*" )
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d + "/*." + extension)
			self.unknown_images[d.split("/")[-1]]=images

		aux = glob.glob( directory + "/Group" )
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d+ "/*." + extension)
			for i in images:
				self.test_images_group.append(i)

	def print_dataset_info(self):
		print("\n\n")
		print("*"*50,"\n*             DATASET INFORMATION                *")
		print("*"*50,"\n")
		print("Number of subjects for training:", self.subjects_number)
		print("Images per subject for training:", len(self.train_images[self.index_to_subject[0]]))
		print("Size of the images to identify: ", self.size)
		print("\n")


	def load_models(self,LOF_name,Annoy_name,train=False):
		if train:

			print("\n\n")
			print("*"*50,"\n*                START TRAINING                  *")
			print("*"*50,"\n")

			if not os.path.exists('models'):
				os.makedirs('models')

			# Known training set

			# Getting the arrays for the training
			Y_train_known_subjects, X_train_known = load_set(self.train_images, self.size)

			# Getting embeddings from FaceNet and normalizing them
			train_known_embeddings = embedder.embeddings(X_train_known)
			X_train_known_embeddings = Normalizer(norm='l2').transform(train_known_embeddings)

			# Encoder in order to associate labels with a certain number
			Y_train_known_index = []
			for subject in Y_train_known_subjects:
				Y_train_known_index.append(self.subject_to_index[subject])

			np.save('models/Y_train_known_index.npy', Y_train_known_index)

			self.Y_train_known_index = np.asarray(Y_train_known_index)

			# Unknown training set

			# Getting the arrays for the training
			Y_unknown_subjects, X_unknown = load_set(self.unknown_images, size = self.size)

			# Getting embeddings from FaceNet and normalizing them
			unknown_embeddings = embedder.embeddings(X_unknown)
			X_unknown_embeddings = Normalizer(norm='l2').transform(unknown_embeddings)

			# Separating training and testing for the unknown
			self.X_unknown,self.Y_unknown_subjects,self.X_unknown_embeddings = shuffle(X_unknown,Y_unknown_subjects,X_unknown_embeddings,random_state=42)

			Y_train_unknown_subjects = []
			X_train_unknown_embeddings = []

			training_percentage = 0.2
			aux = int(training_percentage*len(Y_unknown_subjects))
			counter = 0
			while counter <= aux:
				Y_train_unknown_subjects.append(self.Y_unknown_subjects[-1])
				X_train_unknown_embeddings.append(self.X_unknown_embeddings[-1])
				self.X_unknown=np.delete(self.X_unknown, -1, axis = 0)
				self.Y_unknown_subjects=np.delete(self.Y_unknown_subjects, -1)
				self.X_unknown_embeddings=np.delete(self.X_unknown_embeddings, -1, axis = 0)
				counter = counter + 1

			np.save('models/X_unknown.npy', self.X_unknown)
			np.save('models/Y_unknown_subject.npy', self.Y_unknown_subjects)
			np.save('models/X_unknown_embeddings.npy', self.X_unknown_embeddings)

			X_train_unknown_embeddings = np.array(X_train_unknown_embeddings)

			# Local Outlier Factor

			contamination = len(Y_unknown_subjects)/(len(Y_unknown_subjects) + len(Y_train_known_subjects))

			clf = LocalOutlierFactor(
            		n_neighbors=self.lof_nn,
            		novelty=True,
            		contamination=contamination
        		)
 
			clf.fit( np.vstack((X_train_known_embeddings,X_train_unknown_embeddings)) )
			
			dump(clf, 'models/{}.joblib'.format(LOF_name)) 

			# Annoy 

			vector_length = X_train_known_embeddings.shape[1]

			t = annoy.AnnoyIndex(vector_length,metric="angular")
			
			for i, v in enumerate(X_train_known_embeddings):
				t.add_item(i, v)
				
			t.build(self.Ntrees)
			
			t.save('models/{}.ann'.format(Annoy_name))
			
			# Load Models

			self.LOF_model = load('models/{}.joblib'.format(LOF_name))

			self.annoy_model = annoy.AnnoyIndex(512, metric="angular")
			self.annoy_model.load('models/{}.ann'.format(Annoy_name))

		else:
			
			self.X_unknown = np.load('models/X_unknown.npy')
			self.Y_train_known_index = np.load('models/Y_train_known_index.npy')
			self.Y_unknown_subjects = np.load('models/Y_unknown_subjects.npy')
			self.X_unknown_embeddings = np.load('models/X_unknown_embeddings.npy')
			self.LOF_model = load('models/{}.joblib'.format(LOF_name))

			self.annoy_model = annoy.AnnoyIndex(512, metric="angular")
			self.annoy_model.load('models/{}.ann'.format(Annoy_name))

	def classify_image(self,im):

		emb = embedder.embeddings(np.array([im]))
		emb_im = Normalizer(norm='l2').transform(emb)
		out_or_in = self.LOF_model.predict(emb_im.reshape(1, -1))

		if out_or_in == -1:
			return -1
		elif out_or_in == 1:
			result = self.Y_train_known_index[self.annoy_model.get_nns_by_vector(emb_im.reshape(512, 1), 5,include_distances=False)[0]]
			return result

	def single_image(self,filename):
		print("TESTING SINGLE IMAGE\n")
		im = extract_face(filename,self.size)
		print("Image was classified as",self.index_to_subject[self.classify_image(im)])

	def test_model(self,print_detail=False):

		print("\n\n")
		print("*"*50,"\n*             START TESTING (Known)              *")
		print("*"*50,"\n")

		Y_test_known_subjects, X_test_known = load_set(self.test_images_known, size = self.size)

		Y_real=[]
		for subject in Y_test_known_subjects:
			Y_real.append(self.subject_to_index[subject])
			
		known_prediction = []
		for subject in X_test_known:
			known_prediction.append(self.classify_image(subject))

		if print_detail==True:

			print("\n*************************************************************************")
			print("*                       Testing known images                            *")
			print("*************************************************************************")
			print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
			print("------------------------------|-------------------------------|----------")
			for pred, test in zip(known_prediction,Y_real):
				if pred==-1:
					result="incorrect"
					print("{:<30}| {:<30}| {:<10}".format(self.index_to_subject[test], "* NOT IN DB *", result))
				else:
					if pred==test:
						result="correct"
					else:
						result="incorrect"
					print("{:<30}| {:<30}| {:<10}".format(self.index_to_subject[test],self.index_to_subject[pred], result))

		print("\nKnown accuracy score: ", accuracy_score(Y_real,known_prediction)*100,"%\n\n\n")


		print("\n\n")
		print("*"*50,"\n*             START TESTING (Unknown)             *")
		print("*"*50,"\n")

		unknown_prediction = []
		for subject in self.X_unknown:
			unknown_prediction.append(self.classify_image(subject))


		if print_detail==True:

			print("\n*************************************************************************")
			print("*                       Testing unknown images                            *")
			print("*************************************************************************")
			print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
			print("------------------------------|-------------------------------|----------")
			for pred, subject in zip(unknown_prediction,self.Y_unknown_subjects):
				if pred==-1:
					result="Correct"
					print("{:<30}| {:<30}| {:<10}".format(subject, "* NOT IN DB *", result))
				else:
					result="incorrect"
					print("{:<30}| {:<30}| {:<10}".format(subject,self.index_to_subject[pred], result))

		print("\nKnown accuracy score: ", accuracy_score(np.full(len(unknown_prediction),-1),unknown_prediction)*100,"%\n\n\n")

	def testing_webcam(self, video_path = 0):

		print("TESTING WEBCAM")

		cap = cv2.VideoCapture(video_path)

		color = (246, 181, 100)

		while True:
			ok, frame = cap.read()
			if ok:

				img = frame

				preview = img.copy()

				detections = detector.detect_faces(img)

				if not detections:
					continue
				for detection in detections:
					box = detection['box']

					face = img[abs(box[1]):abs(box[1])+abs(box[3]), abs(box[0]):abs(box[0])+abs(box[2])]

					if self.size == -1:
						face_array = face
					else:
						face_array = cv2.resize(face,(self.size,self.size),interpolation=cv2.INTER_NEAREST)

					identity = self.classify_image(face_array)

					if identity is not None:

						cv2.rectangle(preview,(abs(box[0]), abs(box[1])), (abs(box[0])+abs(box[2]), abs(box[1])+abs(box[3])),color, 1)

						cv2.putText(preview, identity, (abs(box[0]), abs(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,cv2.LINE_AA)

						print(identity)

			cv2.imshow("preview", preview)

			k = cv2.waitKey(0)
			if k == 27:
				break

def load_set(set,size):
	print("\nLoading set...\n")
	images = []
	labels = []
	for person in set:
		for path in set[person]:
			a = extract_face(path,size)
			if a is None:
				print("No face detected in file:",path)
			else:
				images.append(a)
				labels.append(person)
		print("Got",len(set[person]),"images for subject",person)
	print("\n")
	return np.asarray(labels),np.asarray(images)


def extract_face(filename,size):
	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	results = detector.detect_faces(image)
	if not results:
		return None

	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = image[y1:y2, x1:x2]

	if size == -1:
		return face
	else:
		face_array = cv2.resize(face,(size,size),interpolation=cv2.INTER_NEAREST)
		return face_array
