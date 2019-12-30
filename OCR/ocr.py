from PIL import Image
import pytesseract
import argparse
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                help="Type of preprocessing to be done")
args = vars(ap.add_argument())

# Loading the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Check to see if we should apply thresholding to preprocess the image
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
else:
    gray = cv2.medianBlur(gray, 3)

# Write the grayscale image to disk as a temporary file so we can apply OCR to it
filename = "{}.png".format(os.getpgid())
cv2.imwrite(filename=filename, img=gray)

# Loading the image as a PIL/Pillow image, apply OCR, and then delete the
# temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# Show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)