import random
from lib.Pipeline import DataSet
from sklearn.model_selection import ParameterGrid
import numpy as np
import csv


random.seed(42)
vis = False

def grid_search(args_dict):
    grid = ParameterGrid(args_dict)

    f=open("accuracy_info.csv","w+")
    f.write("Slope,Limit,Known Acc,Unknown_acc\n")    
    for params in grid:

        try:
            ds = DataSet(
                directory="datasets/LookDataSet",
                extension="jpg",
                size=24,
                slope_limit=params["slope_limit"],
                intercept_limit=params["intersect_limit"]
            )

            ds.print_dataset_info()
            ds.load_model(name='model',train=False)
            known_acc,unknown_acc=ds.test_model(graphs=False,print_info=False,print_detail=False)
            f.write(str(ds.slope_limit)+","+str(ds.intercept_limit)+","+str(known_acc)+","+str(unknown_acc)+"\n")
            

        except Exception as e:
            print(e)
    f.close()

def main():
    params_grid = {
        "slope_limit": [i for i in np.linspace(0.0,0.5,num=100)],
        "intersect_limit": [i for i in np.linspace(0.0,4.0,num=100)]
    }

    grid_search(params_grid)

if __name__ == "__main__":
    main()
