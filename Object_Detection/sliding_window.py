from helper import pyramid
from helper import sliding_window
import argparse
import time
import cv2


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


# Loading the image and define the window width and height
image = cv2.imread(args["image"])
(width, height) = (128, 128)


# Loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # Loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(width, height)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != width or window.shape[1] != height:
            continue

        # The process use to process the window, such as applying a machine learning classifer to
        # classify contents of the window


        # Draw the windwo
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey()
        time.sleep(0.025)

