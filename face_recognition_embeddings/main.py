from lib.Pipeline import DataSet
import argparse	

# Slope .5
# Intercept .178
# Known Acc 88.44%
# Unknown Acc 81.61%

# Slope .5
# Intercept .19
# Known Acc 90.42%
# Unknown Acc 76.01%

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",			"-dir",	help="Dataset directory",			type=str,	default="datasets/LookDataSet")
	parser.add_argument("--extension",			"-ext",	help="Dataset images extension",	type=str,	default="jpg")
	parser.add_argument("--size",				"-si",	help="Image size",					type=int,	default=24)
	parser.add_argument("--slope_limit",		"-sl",	help="Slope Limit",					type=float, default=.5)
	parser.add_argument("--intercept_limit",	"-il",	help="Intercept Limit",				type=float,	default=.178)
	args = parser.parse_args()
	
	ds = DataSet(
		directory=args.directory,
        extension=args.extension,
        size=args.size,
		slope_limit=args.slope_limit,
		intercept_limit=args.intercept_limit
    )
	
	ds.print_dataset_info()
	ds.load_model(name='model',train=False)
	
	#filename='datasets/LookDataSet/Test/Emma_Watson/Emma_Watson_018_resized.jpg'
	#ds.single_image(filename=filename)
	
	ds.test_model(graphs=False,print_detail=False)
	
	#ds.testing_webcam()


if __name__ == "__main__":
    main()
