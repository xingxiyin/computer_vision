#-*-coding:utf-8-*-
from configs import data_config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from HDF5DatasetWriter import HDF5DatasetWriter
from imutils import paths
import progressbar
import numpy as np
import json
import cv2
import os



# Grab the paths to the images
trainPaths = list(paths.list_images(data_config.IMAGES_PATH))
trainLabels = [path.split(os.path.sep)[-1].split(".")[0] for path in trainPaths]

# Label encoding
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# Perform stratified sampling from the training set to build the testing split from the training data
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(trainPaths, trainLabels, test_size=data_config.NUM_TEST_IMAGES,
                                                                     stratify=trainLabels,
                                                                     random_state=100)


# Perform another stratified sampling, this time to build the validation data
(trainPaths, valPaths, trainLabels, valLabels) = train_test_split(trainPaths, trainLabels, test_size=data_config.NUM_VAL_IMAGES,
                                                                     stratify=trainLabels,
                                                                     random_state=100)

# Construct a list pairing the training, validation, and testing image paths along with
# their conrresponding labels and output HDF5 files
datasets = [
    ("train", trainPaths, trainLabels, data_config.TRAIN_HDF5),
    ("val", valPaths, valLabels, data_config.VAL_HDF5),
    ("test", testPaths, testLabels, data_config.TEST_HDF5)
    ]

# Initialize the image preprocessor and the list of RGB channel average
(R, G, B) = ([], [], [])

# Loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # Create HDF5 weiter
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # Initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # Loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # Load the image and process it
        image = cv2.imread(path)
        # print("Original data shape: ", image.shape)
        image = writer.preprocess(image)
        # print("Cropped data shape: ", image.shape)

        # If we are building the training dataset, then compute the mean of each
        # of each channel in the image, then update the respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # Add the image and label # to the HDFS dataset
        # print("Cropped data shape: ",i, image.shape, label)
        writer.add([image], [label])
        pbar.update(i)

    # Close the HDFS writer
    pbar.finish()
    writer.close()

# Construct a dictionary of average, then serialize the means to a JSON file
print("[INFO] Serializing means...")
D = {"R": np.mean(R), "G":np.mean(G), "B":np.mean(B)}
# print(D)
with open(data_config.DATASET_MEAN, "w") as file:
    file.write(json.dumps(D))
    # file.close()
# file = open(data_config.DATASET_MEAN, "w")
# file.write(json.dumps(D))
# file.close()