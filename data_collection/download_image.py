from imutils import paths
import os
impot cv2
import requests
import argparse


def main(args):
    # Grab the list of url of image
    urls = open(args["urls"]).read().strip().split("\n")

    # Initialize the total number images
    total = 0

    # Loop over the URLs
    for url in urls:
        try:
            # Obtain the image by url
            respon = requests.get(url, timeout=60)

            # Saving the image to disk
            image_path = os.path.sep.join([args["ouput"], "{}.jpg".format(str(total).zfill(8))])
            with open(image_path, "wb") as file:
                file.write(respon.content)

            # Update the counter
            print("[INFO] downloaded: {}".format(image_path))
            total += 1
        except:
            print("[INFO] Error downloading {}... skipping".format(image_path))
            continue

    # Loop over the image paths we just downloaded
    for imagepath in paths.list_images(args["output"]):
        # Initialize the image when it can't be open by the opencv
        delete = False

        try:
            image = cv2.imread(imagepath)

            # if the image is `None` then we could not properly load it
            # from disk, so delete it
            if image is None:
                delete = True
        except:
            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
            print("Except")
            delete = True

        # Check to see if the image should be delted
        if delete:
            print("[INFO] Deleting {}".format(imagepath))
            os.remove(imagepath)


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--urls", required=True,
                    help="path of the file containing image URLs")
    ap.add_argument("-o", "--output", required=True,
                    help="Path to output directory of images")
    args = vars(ap.parse_args())



