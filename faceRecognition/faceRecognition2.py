import functions2 as fn

def main():
    images_per_class        = 60
    # number_person_testing   = 1140
    epsilon                 = 0.01
    height                  = 12*2
    width                   = 10*2
    threshold               = 0.5
    directory               = "lfw/*"
    extension               = "jpg"

    ds = fn.DataSet(
        dir=directory,
        ext=extension,
        images_per_class=images_per_class,
        width=width,
        height=height,
        epsilon=epsilon,
        threshold=threshold
    )


    for test_known in ds.test_images_known:
        test_res = ds.classify(test_known)
        print(test_res)




if __name__ == "__main__":
    main()