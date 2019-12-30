from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

def main():
    # Loading the input image, convert it to a Numpy array, and then
    # reshape it to have an extra dimension
    print("[INFO] Loading the image....")
    image = load_img(args["image"])
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Construct the image generator for data augmentation then
    # initialize the total number of image generated thus far
    aug = ImageDataGenerator(rotation_range=30,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             fill_mode="nearest")


    total = 0

    # Construct the actual python generator
    print("[INFO] generating images....")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                        save_prefix="image", save_format="jpg")

    # Loop over examples from our image data augmentation
    for image in imageGen:
        # Increment counter
        total += 1

        # when reached the specified number of examples, break from the loop
        if total == args["num"]:
            break



if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image", required=True,
                    help="Path of the input images")
    ap.add_argument("-o", "--output", required=True,
                    help="Path of the output image")
    ap.add_argument("--n", "--num", type=int, default=100,
                    help="The number of images we want to generate")
    args = vars(ap.parse_args())

    main(args)