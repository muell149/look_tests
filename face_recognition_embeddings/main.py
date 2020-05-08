from lib.Pipeline import DataSet
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",		  "-dir", help="Dataset directory",        type=str,   default="datasets/LookDataSet")
	parser.add_argument("--extension",        "-ext", help="Dataset images extension", type=str,   default="jpg")
	parser.add_argument("--size",             "-si",  help="Image size",               type=int,   default=24)
	parser.add_argument("--threshold",        "-t",   help="Threshold unknown",        type=float, default=.22)
	args = parser.parse_args()
	
	ds = DataSet(
		directory=args.directory,
        extension=args.extension,
        size=args.size,
		threshold=args.threshold
    )
	
	ds.print_dataset_info()
	ds.load_model(name='model',train=False)
	ds.test_model()


'''
from keras_facenet import FaceNet
import glob
import cv2
from numpy import savez_compressed
from numpy import load
import sys

from random import choice
import numpy as np
from numpy import expand_dims

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import os


sys.exit()
testing_accuracy 		= True
testing_image 			= False

size = 24

if not os.path.exists('Graphs{}'.format(size)):
	os.makedirs('Graphs{}'.format(size))


embedder = FaceNet()

dataset = "datasets/LookDataSet/"
#dataset = "datasets/5-celebrity-faces-dataset/"
print("\n\nLOAD TESTING")
# load test dataset
testX, testy = load_dataset(dataset+'Test/',size=size)
print(testX.shape)

print("\nLOAD TRAINING...")
# load train dataset
trainX, trainy = load_dataset(dataset+'Train/')
print(trainX.shape)

print("\nLOAD UNKNOWN...")

testuX, testuy = load_dataset(dataset+'Unknown/',size=size)
print(testuX.shape)

testX_faces = testX
testuX_faces = testuX
# load the facenet model


# convert each face in the train set to an embedding
newTrainX = embedder.embeddings(trainX)

# convert each face in the test set to an embedding
newTestX  = embedder.embeddings(testX)

# convert each face in the unknown test set to an embedding
newTestuX = embedder.embeddings(testuX)


in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)
testuX = in_encoder.transform(newTestuX)

# label encode targets

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# testing accuracy
if testing_accuracy:
	# predict


	print("\n\nTRAIN")
	yhat_train = model.predict(trainX)
	score_train = accuracy_score(trainy, yhat_train)


	print("\nKNOWN")
	if not os.path.exists('Graphs{}/Known'.format(size)):
		os.makedirs('Graphs{}/Known'.format(size))

	yhat_test = model.predict(testX)
	prob = []
	counter=0
	for prob_vec,class_result in zip(model.predict_proba(testX),yhat_test):
		
		pyplot.plot(range(len(prob_vec)),prob_vec,'ro')
		pyplot.xlabel("Class")
		pyplot.ylabel("Probability")
		pyplot.ylim(0.,.6)
		pyplot.savefig('Graphs{}/Known/plot_class{}_{}.png'.format(size,class_result,counter), dpi=300)
		pyplot.close()
		print(prob_vec[class_result],class_result,np.sum(prob_vec))
		prob.append(prob_vec[class_result])
		counter = counter +1

	score_test = accuracy_score(testy, yhat_test)
	print("max_prob=",max(prob), "min_prob=",min(prob))


	print("\nUNKNOWN")
	if not os.path.exists('Graphs{}/Unknown'.format(size)):
		os.makedirs('Graphs{}/Unknown'.format(size))

	yhatu_test = model.predict(testuX)
	prob = []
	counter =0
	for prob_vec,class_result in zip(model.predict_proba(testuX),yhatu_test):
		pyplot.plot(range(len(prob_vec)),prob_vec,'ro')
		pyplot.xlabel("Class")
		pyplot.ylabel("Probability")
		pyplot.ylim(0.,.6)
		pyplot.savefig('Graphs{}/Unknown/plot_class{}_{}.png'.format(size,class_result,counter), dpi=300)
		pyplot.close()
		print(prob_vec[class_result],class_result, np.sum(prob_vec))
		prob.append(prob_vec[class_result])
		counter=counter+1
	print("max_prob=",max(prob), "min_prob=",min(prob))

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
'''



if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------------------------------------
