from lib.Pipeline import DataSet
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--directory",		  "-dir", help="Dataset directory",        type=str,   default="datasets/LookDataSet")
	parser.add_argument("--extension",        "-ext", help="Dataset images extension", type=str,   default="jpg")
	parser.add_argument("--size",             "-si",  help="Image size",               type=int,   default=24)
	parser.add_argument("--threshold",        "-t",   help="Threshold unknown",        type=float, default=.7)
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

if __name__ == "__main__":
    main()
