from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montage
from imutils import paths
import numpy as np
import argparse
import random
import cv2

# Construct the argument parser and the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="Path to the image")
ap.add_argument("-m", "--model", required=True,
                help="Path to pre-trianed model")
args = vars(ap.parse_args())


# Loading the pre-trained model
print("[INFO] Loading pre-trainded model...")
model = load_model(args["model"])

# load our original input image
orig = cv2.imread(args["images"])

# pre-process our image by converting it from BGR to RGB channel
# ordering (since our Keras mdoel was trained on RGB ordering),
# resize it to 64x64 pixels, and then scale the pixel intensities
# to the range [0, 1]
image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0

# order channel dimensions (channels-first or channels-last)
# depending on our Keras backend, then add a batch dimension to the image
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# make predictions on the input image
pred = model.predict(image)
pred = pred.argmax(axis=1)[0]

# an index of zero is the 'parasitized' label while an index of
# one is the 'uninfected' label
label = "Parasitized" if pred == 0 else "Uninfected"
color = (0, 0, 255) if pred == 0 else (0, 255, 0)

# resize our original input (so we can better visualize it) and
# then draw the label on the image
orig = cv2.resize(orig, (128, 128))
cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show the output image
cv2.imshow("Image", output_image)
cv2.waitKey(0)
