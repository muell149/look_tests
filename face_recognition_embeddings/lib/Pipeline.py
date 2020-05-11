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
		print("Images per subject for training:", len(self.train_images[self.index_to_subject[0]]))
		print("Size of the images to identify: ", self.size)
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
		print("*"*50,"\n*             START TESTING (Known)              *")
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
		print("Accuracy on known:",score_test_known*100,"%\n\n")

		print("\n\n")
		print("*"*50,"\n*            START TESTING (Unknown)             *")
		print("*"*50,"\n")
		test_y_subjects, test_x = load_set(self.test_images_unknown, size = self.size)

		test_embeddings = embedder.embeddings(test_x)
		test_x = Normalizer(norm='l2').transform(test_embeddings)

		test_y = [-1 for i in range(len(test_y_subjects))]

		test_y = np.asarray(test_y)

		y_test_pred = self.model.predict(test_x)
		y_test_proba = self.model.predict_proba(test_x)
		y_aux=[]
		for pred, proba in zip(y_test_pred,y_test_proba):
			y_aux.append(identify_unknown(proba,pred,self.threshold))

		score_test_unknown = accuracy_score(test_y,y_aux)
		print("Accuracy on unknown:",score_test_unknown*100,"%\n\n")

		return score_test_known*100, score_test_unknown*100

	def classify_image(self,im):
		emb_im = embedder.embeddings(np.array([im]))

		pred = self.model.predict(emb_im)
		pred_proba = self.model.predict_proba(emb_im)

		result = identify_unknown(pred_proba[0], pred[0], self.threshold)

		if result == -1:
			return "Unknown"
		else:
			return self.index_to_subject[result] 

	def single_image(self,filename):
		print("TESTING SINGLE IMAGE\n")
		im = extract_face(filename,self.size)
		print("Image was classified as",self.classify_image(im))

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


def identify_unknown(x,index,t):
	limit=t*x[index]
	new_x=np.delete(x,index)
	for i in new_x:
		if i>=limit:
			ind = -1
			break
		else:
			ind = index
	return ind

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

	if size == -1:
		return face
	else:
		face_array = cv2.resize(face,(size,size),interpolation=cv2.INTER_NEAREST)
		return face_array