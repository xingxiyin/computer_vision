#-*-coding:utf-8-*-
from configs import data_config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import progressbar
import numpy as np
import hdfs
import h5py
import json
import cv2
import os

def HDF5DatasetWriter(datashape, outputPath, dataKey="image", bufSize=1000):
    # Check if the ouput path exists or not. If existed, raise an exception
    if os.path.exists(outputPath):
        raise ValueError("The supplied 'outputPath' already 'exists and cannot be overwriten Manually delete the file before continuing", outputPath)

    # Open teh HDF5 database for writting and create two datasets:
    # One to store the images/features and another to store the class labels
    db = h5py.File(outputPath, "w")
    data = db.create_dataset(dataKey, datashape, dtype="float")
    labels = db.create_dataset("labels", (datashape[0],), dtype="int")

    # Store the buffer size, the initialize the buffer itself along with the index
    # into the dataset
    buffer = {"data":[], "labels":[]}



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
    ("test", testPaths, testLabels, data_config.TEST_HDF5)]

# Initialize the image preprocessor and the list of RGB channel average

(R, G, B) = ([], [], [])