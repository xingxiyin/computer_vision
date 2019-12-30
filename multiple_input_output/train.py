import matplotlib
matplotlib.use("Agg")

from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from fashionnet import FashionNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	            help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	            help="path to output model")
ap.add_argument("-l", "--categorybin", required=True,
	            help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	            help="path to output color label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output",
	            help="base filename for generated plots")
args = vars(ap.parse_args())


# Initialize the number of epochs to train for, initial learning rate, batch size and image dimensions
EPOCH = 50
INIT_LR = 1e-3
Batch_size = 32
image_dims = (96, 96, 3)

# Grab the image paths and randomly shuffle them
print("[INFO] Loading images...")
imagePaths = sorted(list(paths.list_images(args["datasize"])))
random.seed(42)
random.shuffle(imagePaths)

# Initialize the data, clothing category labels, along with the color labels
data = []
categoryLabels = []
colorLabels = []

# Loop over the input images
for imagePath in imagePaths:
    # Loading the image, pre-process it and store in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    data.append(image)

    # Extract the clothing color and category from the path and update the repective lists
    (color, cat) = imagePath.split(os.path.sep)[-2].split("_")
    categoryLabels.append(cat)
    colorLabels.append(color)


# Scale the raw pixel instensities to the range [0, 1] and convert to a Numpy array
data = np.array(data, dtype="float")/255.0
print("[INFO] Data matrix: {} images ({:0.2f}MB)".format(len(imagePaths), data.nbytes/(1024*1000.0)))

# Convert the label lists to Numpy arrays prior to binarization
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

# Binarize both sets of labels
print("[INFO] Binarizing labels...")
categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)

# Partition  the data into training and testing splits using 80% of the data
# for training and the remaining 20% for testing
split = train_test_split(data, categoryLabels, colorLabels, test_size=0.3, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = split

# Initialize the FashionNet multi-output network
model = FashionNet.build(width=96, height=96,
                         numCategories=len(categoryLB.classes_),
                         numColors=len(colorLB.classes_),
                         finalAct="softmax")

# Define two dictionaries: one that specifies the loss method for
# each ouput of the network along with a second dictionary that
losses = {"category_output": "categorical_crossentropy",
          "color_output": "categorical_crossentropy"}
lossWeights = {"category_output": 1.0, "color_output": 1.0}

# Initialize the optimizer and compile the model
print("[INFO] Compiling the model...")
optimizer = Adam(lr=INIT_LR, decay=INIT_LR/EPOCH)
model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])

# Train the network to perform multi-output classification
H = model.fit(trainX,
              {"category_output":trainCategoryY, "color_output":trainColorY},
              validation_data=(testX, {"category_output":testCategoryY, "color_output":testColorY}),
              epochs=EPOCH,
              verbose=1)

# Saving the model to disk
print("[INFO] Serializing network...")
model.save(args["model"])

# Saving the category binarizer to disk
print("[INFO] Saving the category binarizer to disk")
f = open(args["categorybin"], "wb")
f.write(pickle.dumps(categoryLB))
f.close()

# Saving the color binarizer to disk
print("[INFO] Serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()


# plot the total loss, category loss, and color loss
lossNames = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()



# create a new figure for the accuracies
accuracyNames = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()