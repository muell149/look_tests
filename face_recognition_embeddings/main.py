from lib.Pipeline import DataSet
import argparse	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",			"-dir",	help="Dataset directory",			type=str,	default="datasets/LookDataSet")
	parser.add_argument("--extension",			"-ext",	help="Dataset images extension",	type=str,	default="jpg")
	parser.add_argument("--size",				"-si",	help="Image size",					type=int,	default=24)
	parser.add_argument("--lof_nn",    "-lof_nn", help="Local Outlier Factor Nearest Neighbour Number", type=int, default=13)
	parser.add_argument("--ntrees",   "-nt", help="Number of trees Annoy", type=int, default=10)
	args = parser.parse_args()
	
	ds = DataSet(
		directory=args.directory,
        extension=args.extension,
        size=args.size,
		lof_nn=args.lof_nn,
		Ntrees=args.ntrees
    )
	
	ds.print_dataset_info()
	ds.load_models(LOF_name='lof1',Annoy_name='annoy1',train=True)
	
	#filename='datasets/LookDataSet/Test/Emma_Watson/Emma_Watson_018_resized.jpg'
	#ds.single_image(filename=filename)
	
	ds.test_model(print_detail=False)
	
	#ds.testing_webcam()


if __name__ == "__main__":
    main()
