from keras.applications import  ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
                help="Name of pre-trained network to use")
args = vars(ap.parse_args())


# Define a dictionary that maps names to their classes inside Keras
MODELS = {"vgg16": VGG16,
          "vgg19":VGG19,
          "inception": InceptionV3,
          "xception":Xception,
          "resnet":ResNet50}

# Ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be"
                         "a key in the 'MODELS' dictionary")


# VGG16, VGG19, and ResNet all accept 224*224 input images while Inception V3 and
# Xception require 229*229 pixel inputs


# Initialize the input image shape (224*224 pixels) along with the pre-processing
# function (which might need to be changed based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# If we are using the InceptionV3 or Xception networks, then we need to set the
# input shape to (299*299) [rather than (224*224)]
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


# Loading the network weights from disk
print("[INFO] Loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# Loading the input image using the Keras helper utility while ensuring
# the image is resized to "inputShape", the required input dimensions for
# the ImageNet pre-trained network
print("[INFO] Loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)


# Our input image is now represented as a Numpy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)

# Pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean substraction, scaling, etc.)
image = preprocess(image)

# Classify the image
print("[INFO] Classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# Loop over the predictions and display the rank-5 predictions + probablities
# to our terminal
for (index, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}:{:0.2f}%".format(index, label, prob*100))


# Loading the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob*100), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2)
cv2.imshow("Classiifcation", orig)
cv2.waitKey(0)
