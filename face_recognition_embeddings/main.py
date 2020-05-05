from lib.Pipeline import load_dataset, load_faces
from keras_facenet import FaceNet
import glob
import cv2
from numpy import savez_compressed
from numpy import load
import sys

from random import choice
from numpy import load
from numpy import expand_dims

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

testing_accuracy 		= True
testing_image 			= True


embedder = FaceNet()

dataset = "datasets/LookDataSet/"
#dataset = "datasets/5-celebrity-faces-dataset/"
print("\n\nLOAD TESTING")
# load test dataset
testX, testy = load_dataset(dataset+'Unknown/',size=30)
print(testX.shape)

print("\nLOAD TRAINING...")
# load train dataset
trainX, trainy = load_dataset(dataset+'Train/')
print(trainX.shape)

testX_faces = testX

# load the facenet model


# convert each face in the train set to an embedding
newTrainX = embedder.embeddings(trainX)

# convert each face in the test set to an embedding
newTestX  = embedder.embeddings(testX)



in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)

# label encode targets
print("here2")
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
print("here1")
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# testing accuracy
if testing_accuracy:
	# predict
	print("here")
	yhat_train = model.predict(trainX)
	yhat_test = model.predict(testX)
	print("Finish")
	# score
	score_train = accuracy_score(trainy, yhat_train)
	score_test = accuracy_score(testy, yhat_test)
	# summarize
	print('\n\nAccuracy: train=%.3f, test=%.3f\n' % (score_train*100, score_test*100))

if testing_image:
	selection = choice([i for i in range(testX.shape[0])])
	random_face_pixels = testX_faces[selection]
	random_face_emb = testX[selection]
	random_face_class = testy[selection]
	random_face_name = out_encoder.inverse_transform([random_face_class])
	# prediction for the face
	samples = expand_dims(random_face_emb, axis=0)
	yhat_class = model.predict(samples)
	yhat_prob = model.predict_proba(samples)
	# get name
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	print('Expected: %s' % random_face_name[0])
	# plot for fun
	pyplot.imshow(random_face_pixels)
	title = '%s (%.3f)' % (predict_names[0], class_probability)
	pyplot.title(title)
	pyplot.show()