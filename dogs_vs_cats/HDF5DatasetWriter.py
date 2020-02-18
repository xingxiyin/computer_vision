#-*-coding:utf-8-*-
import h5py
import imutils
import cv2
import os

class HDF5DatasetWriter:
    def __init__(self, datashape, outputPath, dataKey="image", bufSize=100):
        # Check if the ouput path exists or not. If existed, raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already 'exists and cannot be overwriten Manually delete the file before continuing", outputPath)

        # Open teh HDF5 database for writting and create two datasets:
        # One to store the images/features and another to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, datashape, dtype="float")
        self.labels = self.db.create_dataset("labels", (datashape[0],), dtype="int")

        # Store the datashape
        self.width, self.height = datashape[1], datashape[2]
        # print("self.width, self.height ", self.width, self.height)
        # Store the buffer size, the initialize the buffer itself along with the index
        # into the dataset
        self.bufSize = bufSize
        self.buffer = {"data":[], "labels":[]}
        self.idx = 0

    def preprocess(self, image):
        # Grab the dimensions of the image and then initialize the deltas to use when cropping
        (height, width) = image.shape[:2]
        delta_width = 0
        delta_height = 0

        # If the width is smaller than the height, then resize along the width
        # (i.e the smaller dimension) and then update the deltas to crop
        # the desired dimension
        if width < height:
            image = imutils.resize(image, width=self.width, inter=cv2.INTER_AREA)
            delta_height = int((image.shape[0] - self.height)/2.0)
        else:
            # Otherwise, the height is smaller than the width so resize along the
            # height and then update the deltas to crop along the width
            image = imutils.resize(image, height=self.height, inter=cv2.INTER_AREA)
            delta_width = int((image.shape[1] - self.width)/2.0)

        # Now that our images have been resized, we need to re-grab the width and heigth
        # followed by performing the crop
        (height, width) = image.shape[:2]
        image = image[delta_height:height-delta_height, delta_width:width-delta_width]

        # Finally, resize the image to the provided spatial dimensions to ensure our output
        # image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def add(self, rows, labels):
        # Adding the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if th buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # Write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data":[], "labels":[]}

    def storeClassLables(self, classLabels):
        # Create a dataset to store the actual class lavel names, then store the clas lael
        dt = h5py.special_dtype(vlen=unicode)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # Check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # Close the dataset
        self.db.close()