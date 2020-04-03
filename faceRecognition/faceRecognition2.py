import cv2
import random

from functions2 import DataSet


def main():
    directory               = "datasets/lfw/*"
    extension               = "jpg"
    images_per_class        = 50
    width                   = 10*2
    height                  = 12*2
    vertical                = 4                     # Has to be an even number
    horizontal              = 2                     # Has to be an even number
    epsilon                 = 0.01
    threshold               = 0.1

    ds = DataSet(
        dir=directory,
        ext=extension,
        images_per_class=images_per_class,
        width=width,
        height=height,
        vertical=vertical,
        horizontal=horizontal,
        epsilon=epsilon,
        threshold=threshold,
        vis=False
    )

    print("Testing known images")
    print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
    print("------------------------------|-------------------------------|----------")
    for i, test_known in enumerate(ds.test_images_known):
        test_res = ds.classify(test_known)
        result = "correct" if test_res == i else "incorrect"
        if test_res == -1:
            class_result = "* UNKNOWN *"
        else:
            class_result = ds.classes[test_res]
        print("{:<30}| {:<30}| {:<10}".format(ds.classes[i], class_result, result))

    # print("Testing unknown images")
    # for test_unknown in random.sample(ds.test_images_unknown, k=ds.number_classes):
    #     test_res = ds.classify(test_unknown, plot=False)
    #     print(test_res)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
