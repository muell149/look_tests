import cv2
import random
import argparse
from lib.Pipeline import detector
from functions import DataSet
import sys
import datetime


random.seed(13)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",        "-dir", help="Dataset directory",        type=str,   default="datasets/LookDataSet")
    parser.add_argument("--extension",        "-ext", help="Dataset images extension", type=str,   default="jpg")
    parser.add_argument("--images_per_class", "-ipc", help="Images to use per class",  type=int,   default=14)
    parser.add_argument("--size",             "-si",  help="Image size",               type=int,   default=30)
    parser.add_argument("--vertical",         "-ve",  help="Vertical splits",          type=int,   default=2)
    parser.add_argument("--horizontal",       "-ho",  help="Horizontal splits",        type=int,   default=2)
    parser.add_argument("--epsilon",          "-e",   help="Epsilon",                  type=float, default=0.0)
    parser.add_argument("--threshold",        "-t",   help="Classification threshold", type=float, default=0.0001)
    parser.add_argument("--vis",              "-v",   help="Show aligned and crop images", type=bool,default=False)
    args = parser.parse_args()

    ds = DataSet(
        dir=args.directory,
        ext=args.extension,
        images_per_class=args.images_per_class,
        size=args.size,
        vertical=args.vertical,
        horizontal=args.horizontal,
        epsilon=args.epsilon,
        threshold=args.threshold,
        vis=args.vis
    )

    vis = False

    print("\n*************************************************************************")
    print("*                       Testing known images                            *")   
    print("*************************************************************************")
    print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
    print("------------------------------|-------------------------------|----------")

    counter = 0
    i = 0
    begin_time = datetime.datetime.now()
    for subject in ds.test_images_known:
        for image in ds.test_images_known[subject]:
            test_res = ds.classify(image, vis=vis)
            counter = counter + 1
            if test_res==-1:
                result = "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject, "* NOT IN DB *", result))
            elif test_res is None:
                print("{:<30}| {:<30}|".format(subject, "* NO FACE FOUND *"))
            else:
                if ds.classes[test_res]==subject:
                    result = "correct"
                    i=i+1  
                else:
                    result = "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject,ds.classes[test_res], result))

    print("Accuracy:", i*100./counter)
    print("Average time per person:", (datetime.datetime.now() - begin_time)/ counter)


    counter = 0
    i = 0
    print("\n\n*************************************************************************")
    print("*                      Testing unknown images                           *")   
    print("*************************************************************************")
    print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
    print("------------------------------|-------------------------------|----------")
    for subject in ds.test_images_unknown:
        for image in ds.test_images_unknown[subject]:
            test_res = ds.classify(image, vis=vis)
            counter = counter + 1
            if test_res==-1:
                result = "correct"
                i=i+1
                print("{:<30}| {:<30}| {:<10}".format(subject, "* NOT IN DB *", result))
            elif test_res is None:
                print("{:<30}| {:<30}|".format(subject, "* NO FACE FOUND *"))
            else:
                result = "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject,ds.classes[test_res], result))
    print("Accuracy:", i*100./counter)

    print("\n\n*************************************************************************")
    print("*                       Testing group images                            *")   
    print("*************************************************************************\n")

    for photo_path in ds.test_images_group:
        
        photo = cv2.imread(photo_path)

        preview = photo.copy()

        detections = detector.detect_face(photo)

        if detections is None:
            continue                             # Return None if no face is found

        boxes = detections[0]
        points = detections[1]

        preview = photo.copy()

        chips = detector.extract_image_chips(photo,points)

        for chip,box in zip(chips,boxes):

            id = ds.classify(chip, vis=vis)
            if id ==-1:
                class_name = "Not in DS"
            elif id == None:
                class_name = "No detection"
            elif id != -1:
                class_name = ds.classes[id]

            if id is not None:
                cv2.rectangle(preview,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(246, 181, 100), 1)
                cv2.putText(preview, str(class_name), (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (246, 181, 100), 1,cv2.LINE_AA)

                max_x = preview.shape[1]
                max_y = preview.shape[0]

                im_boundary = cv2.resize(chip,(24,24),interpolation = cv2.INTER_AREA)

                if max_x-int(box[2])<=int(box[0]) and max_y-int(box[3])<=int(box[1]):
                    preview[int(box[1])-im_boundary.shape[0]:int(box[1]),int(box[0])-im_boundary.shape[1]:int(box[0]),:] = im_boundary

                elif max_x-int(box[2])>=int(box[0]) and max_y-int(box[3])>=int(box[1]):
                    preview[int(box[3]):int(box[3])+im_boundary.shape[0],int(box[2]):int(box[2])+im_boundary.shape[1],:] = im_boundary

                elif max_x-int(box[2])<=int(box[0]) and max_y-int(box[3])>=int(box[1]):
                    preview[int(box[3]):int(box[3])+im_boundary.shape[0],int(box[0])-im_boundary.shape[1]:int(box[0]),:] = im_boundary

                elif max_x-int(box[2])>=int(box[0]) and max_y-int(box[3])<=int(box[1]):
                    preview[int(box[1])-im_boundary.shape[0]:int(box[1]),int(box[2]):int(box[2])+im_boundary.shape  [1],:] = im_boundary

        cv2.imshow("preview", preview)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()