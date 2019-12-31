from helper import pyramid
from skimage.transform import pyramid_gaussian
import argparse
import cv2


# Construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.05, help="Scale factor size")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])

# Method 1: No smooth, just scaling
# Loop over the image pyramid
for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
    # show the resized image
    cv2.imshow("Layer {}".format(i+1), resized)
    cv2.waitKey(0)

# Close all the windows
cv2.destroyAllWindows()



# Method 2: Resizing + Gaussian Smoothing
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
    # if the image is too small, break from the loop
    if resized.shape[0] < 30 or resized.shape[1] < 30:
        break

    # Show the resized image
    cv2.imshow("Layer {}".format(i+1), resized)
    cv2.waitKey(0)



"""
If you are using the HOG descriptor for object classification you’ll want to use the
first method since smoothing tends to hurt classification performance.

If you are trying to implement something like SIFT or the Difference of Gaussian keypoint 
detector, then you’ll likely want to utilize the second method (or at least incorporate 
smoothing into the first).
"""