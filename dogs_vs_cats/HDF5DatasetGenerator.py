#-*-coding:utf-8-*-
from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessor=None, aug=None,
                 binarize=True, classes=2):
        """

        :param dbPath: The path to the HDF5 dataset which stores the images and labels
        :param batchSize: The size of mini-batches to yield when training
        :param preprocessor: The list of image preprocessors we are going to apply
        :param aug: Keras ImageDataGenerator
        :param binarize: Whether or not the one-hot encoded needs to take place
        :param classes: The number of unique class labels in the dataset
        """
        # Store the batch size, preprocessors, and data augmentor, whether
        # or not the labels should be binarized, along with the total number of classes
        self.batchSize = batchSize
        self.preprocessor = preprocessor
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # Open the HDF5 database for reading and determine the total number of
        # enteries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["image"].shape[0]

    def generator(self, passes=np.inf):
        """

        :param pasess: The total number of epochs
        :return:
        """
        # Initialize the epoch count
        epochs = 0

        # Keep looping infinitely -- the model will stop once we have reach the desired number of epochs
        while epochs < passes:
            # Loop over each batch of data points in the HDF5 dataset
            for index in np.arange(0, self.numImages, self.batchSize):
                # Extract the images and labels from the HDF5 dataset
                images = self.db["image"][index: index + self.batchSize]
                labels = self.db["labels"][index: index + self.batchSize]

                # Check to see if th labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # Check to see if our preprocessor are not None
                if self.preprocessor is not None:
                    # Initialize the list of processed images
                    procImages = []

                    # Loop over the images
                    for image in images:
                        # Loop over the preprocessors and apply each to the image
                        for processor in self.preprocessor:
                            image = processor.preprocess(image)

                        # Update the list of processd image
                        procImages.append(image)

                    # Update the images array to be the processed images
                    images = np.array(procImages)

                # If the data augmentor exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                # Yield a tuple of images and labels
                yield (images, labels)

            # Increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()







