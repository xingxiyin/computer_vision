import config
from imutils import paths
import random
import shutil
import os

# Grab the paths to all input images in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(100)
random.shuffle(imagePaths)

# Compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# Using part of the training dataa for validation
val_num = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# Define the datasets that we'll be building
datasets = [("training", trainPaths, config.TRAIN_PATH),
            ("validation", valPaths, config.VALIDATION_PATH),
            ("testing", testPaths, config.TEST_PATH)]


# Loop over the datasets
for (dataType, imagePaths, baseOutput) in datasets:
    # Show which dataset we are creating
    print("[INFO] Buildint {} split.".format(dataType))

    # If the output base output directory dose not exist, then create it
    if not os.path.exists(baseOutput):
        print("[INFO] Creating {} directory".format(baseOutput))
        os.makedirs(baseOutput)

    # Loop over the input image paths
    for inputPath in imagePaths:
        # Extract the filename and the class label of the input image
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]

        # Build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])

        # If the label ouput directory dose not exist, creat it
        if not os.path.exists(labelPath):
            print("[INFO] Creating {} directory".format(labelPath))
            os.makedirs(labelPath)

        # Construct the path to the destination image and then copy the image itself
        path = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, path)