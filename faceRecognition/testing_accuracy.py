import cv2
import random
import argparse

from functions import DataSet
import sys


random.seed(13)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",        "-dir", help="Dataset directory",        type=str,   default="datasets/LookDataSet2")
    parser.add_argument("--extension",        "-ext", help="Dataset images extension", type=str,   default="jpg")
    parser.add_argument("--images_per_class", "-ipc", help="Images to use per class",  type=int,   default=14)
    parser.add_argument("--size",             "-si",  help="Image size",               type=int,   default=24)
    parser.add_argument("--vertical",         "-ve",  help="Vertical splits",          type=int,   default=4)
    parser.add_argument("--horizontal",       "-ho",  help="Horizontal splits",        type=int,   default=2)
    parser.add_argument("--epsilon",          "-e",   help="Epsilon",                  type=float, default=0.0)
    parser.add_argument("--threshold",        "-t",   help="Classification threshold", type=float, default=0.22)
    parser.add_argument("--vis",              "-v",   help="Show aligned and crop images", type=bool,default=False)
    args = parser.parse_args()
    vis = False

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

    print("\nTesting known images\n")
    print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
    print("------------------------------|-------------------------------|----------")
    for subject in ds.test_images_known:
        for image in ds.test_images_known[subject]:
            test_res = ds.classify(image, vis=vis)

            if test_res==-1:
                result = "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject, "* NOT IN DB *", result))
            elif test_res is None:
                print("{:<30}| {:<30}|".format(subject, "* NO FACE FOUND *"))
            else:
                result = "correct" if ds.classes[test_res]==subject else "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject,ds.classes[test_res], result))

    print("\n\nTesting unknown images\n")
    print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
    print("------------------------------|-------------------------------|----------")
    for subject in ds.test_images_unknown:
        for image in ds.test_images_unknown[subject]:
            test_res = ds.classify(image, vis=vis)

            if test_res==-1:
                result = "correct"
                print("{:<30}| {:<30}| {:<10}".format(subject, "* NOT IN DB *", result))
            elif test_res is None:
                print("{:<30}| {:<30}|".format(subject, "* NO FACE FOUND *"))
            else:
                result = "incorrect"
                print("{:<30}| {:<30}| {:<10}".format(subject,ds.classes[test_res], result))

    print("\n\nTesting group images\n")

    for photo in ds.test_images_group:
        print(photo)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()