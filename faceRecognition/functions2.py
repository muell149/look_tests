import glob
import random
import numpy as np

from lib.Pipeline import detect_and_align
from sklearn.preprocessing import normalize


class DataSet:
    def __init__(self, dir, ext, images_per_class, height, width):
        # images_subjects = []

        self.test_images_known = []
        self.test_images_unknown = []

        A = []
        for directory in glob.glob(dir):
            images = glob.glob(directory + "/*." + ext)

            if len(images) > images_per_class:
                subject_images_sample = random.sample(images, k=images_per_class+1)
                self.test_images_known.append(subject_images_sample.pop())

                aligned_images = [detect_and_align(im, width, height) for im in subject_images_sample]
                if any(a is None for a in aligned_images):
                    continue
                else:
                    print("{:<50} | {:<5}".format(directory.split("/")[-1], len(images)))
                    for a in aligned_images:
                        A.append(a.flatten("F"))

        self.matrix = normalize(np.asmatrix(A).T, axis=0, norm="l2")
