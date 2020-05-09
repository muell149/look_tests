from lib.Pipeline import DataSet, detector,extract_face
import argparse
import cv2

video_path = 0
cap = cv2.VideoCapture(video_path)

color = (246, 181, 100)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",		  "-dir", help="Dataset directory",        type=str,   default="datasets/LookDataSet")
	parser.add_argument("--extension",        "-ext", help="Dataset images extension", type=str,   default="jpg")
	parser.add_argument("--size",             "-si",  help="Image size",               type=int,   default=24)
	parser.add_argument("--threshold",        "-t",   help="Threshold unknown",        type=float, default=.5)
	args = parser.parse_args()
	
	ds = DataSet(
		directory=args.directory,
        extension=args.extension,
        size=args.size,
		threshold=args.threshold
    )
	
	ds.print_dataset_info()
	ds.load_model(name='model',train=False)
	#ds.test_model()
	
	# print("TESTING SINGLE IMAGE\n")

	# filename='datasets/LookDataSet/Test/Emma_Watson/Emma_Watson_018_resized.jpg'
	# im = extract_face(filename,ds.size)
	# print("Image was classified as",ds.classify_image(im))

	print("TESTING WEBCAM")

	while True:
		ok, frame = cap.read()
		if ok:

			img = frame

			preview = img.copy()

			detections = detector.detect_faces(img)
			
			if not detections:
				continue
			for detection in detections:
				x1, y1, width, height = detection['box']
				x1, y1 = abs(x1), abs(y1)
				x2, y2 = x1 + width, y1 + height

				face = img[y1:y2, x1:x2]

				if ds.size == -1:
					face_array = face
				else:
					face_array = cv2.resize(face,(ds.size,ds.size),interpolation=cv2.INTER_NEAREST)

				identity = ds.classify_image(face_array)

				if identity is not None:
					
					cv2.rectangle(preview,(x1, y1), (x2, y2),color, 1)

					cv2.putText(preview, identity, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,cv2.LINE_AA)

					# max_x = preview.shape[1]
					# max_y = preview.shape[0]

					# im_boundary = cv2.resize(chip,(24,24),interpolation = cv2.INTER_AREA)

					# if max_x-int(box[2])<=int(box[0]) and max_y-int(box[3])<=int(box[1]):
					#     preview[int(box[1])-im_boundary.shape[0]:int(box[1]),int(box[0])-im_boundary.shape[1]:int(box[0]),:] = im_boundary

					# elif max_x-int(box[2])>=int(box[0]) and max_y-int(box[3])>=int(box[1]):
					#     preview[int(box[3]):int(box[3])+im_boundary.shape[0],int(box[2]):int(box[2])+im_boundary.shape[1],:] = im_boundary

					# elif max_x-int(box[2])<=int(box[0]) and max_y-int(box[3])>=int(box[1]):
					#     preview[int(box[3]):int(box[3])+im_boundary.shape[0],int(box[0])-im_boundary.shape[1]:int(box[0]),:] = im_boundary

					# elif max_x-int(box[2])>=int(box[0]) and max_y-int(box[3])<=int(box[1]):
					#     preview[int(box[1])-im_boundary.shape[0]:int(box[1]),int(box[2]):int(box[2])+im_boundary.shape  [1],:] = im_boundary
					
					print(identity)
				
		cv2.imshow("preview", preview)
		
		k = cv2.waitKey(0)
		if k == 27:
			break




if __name__ == "__main__":
    main()
