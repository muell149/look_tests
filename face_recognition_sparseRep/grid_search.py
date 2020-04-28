import cv2
import random

from functions2 import DataSet
from sklearn.model_selection import ParameterGrid

random.seed(42)
vis = False

def grid_search(args_dict):
    grid = ParameterGrid(args_dict)
    for params in grid:
        print("\n"+"*"*50+"\n")
        print(params)

        try:
            ds = DataSet(
                dir="datasets/lfw/*",
                ext="jpg",
                images_per_class=params["images_per_class"],
                width=params["width"],
                height=params["height"],
                vertical=params["vertical"],
                horizontal=params["horizontal"],
                epsilon=params["epsilon"],
                threshold=params["threshold"],
                vis=vis
            )

            known_correct = 0
            for i, test_known in enumerate(ds.test_images_known):
                test_res = ds.classify(test_known, vis=vis)
                if test_res == i:
                    known_correct += 1

            known_score = known_correct/len(ds.test_images_known)
            print("KNOWN SCORE:", known_score)


            unknown_correct = 0
            if len(ds.test_images_unknown) >= len(ds.test_images_known):
                unknown_list = random.sample(ds.test_images_unknown, k=ds.number_classes)
            else:
                unknown_list = ds.test_images_unknown

            for test_unknown in unknown_list:
                test_res = ds.classify(test_unknown, plot=False, vis=vis)
                if test_res == -1:
                    unknown_correct+=1

            unknown_score = unknown_correct/len(unknown_list)
            print("UNKNOWN SCORE:", unknown_score)


            cv2.destroyAllWindows()

        except Exception as e:
            print(e)


def main():
    params_grid = {
        "images_per_class": [15, 20, 25],
        "width": [10, 20, 30],
        "height": [12, 24, 36],
        "vertical": [2, 4],
        "horizontal": [2],
        "epsilon": [0.0],
        "threshold": [0.1, 0.2, 0.3]
    }

    grid_search(params_grid)

if __name__ == "__main__":
    main()
