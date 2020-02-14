#-*-coding:utf-8-*-
from keras.applications import VGG16
import argparse

# Constcut the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=-1,
                help="Whether or not to include top of CNN")
args = vars(ap.parse_args())

# Loading the VGG-16 network
print("[INFO] Loaidng VGG-16 network...")
model = VGG16(weights="imagenet", include_top=args["include_top"]>0)
print("[INFO] Showing layers...")

# Loop over the layers in the network and display them to the console
for (index, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(index, layer.__class__.__name__))