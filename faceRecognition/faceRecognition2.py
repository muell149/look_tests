import functions2 as fn

def main():
    directory               = "datasets/lfw/*"
    extension               = "jpg"
    images_per_class        = 60
    width                   = 10*2
    height                  = 12*2
    epsilon                 = 0.01
    threshold               = 0.1

    ds = fn.DataSet(
        dir=directory,
        ext=extension,
        images_per_class=images_per_class,
        width=width,
        height=height,
        epsilon=epsilon,
        threshold=threshold,
        vis=True
    )


    for test_known in ds.test_images_known:
        test_res = ds.classify(test_known, plot=False)
        print(test_res)




if __name__ == "__main__":
    main()
