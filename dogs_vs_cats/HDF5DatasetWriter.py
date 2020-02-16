#-*-coding:utf-8-*-
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, datashape, outputPath, dataKey="image", bufSize=1000):
        # Check if the ouput path exists or not. If existed, raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already 'exists and cannot be overwriten Manually delete the file before continuing", outputPath)

        # Open teh HDF5 database for writting and create two datasets:
        # One to store the images/features and another to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, datashape, dtype="float")
        self.labels = self.db.create_dataset("labels", (datashape[0],), dtype="int")

        # Store the buffer size, the initialize the buffer itself along with the index
        # into the dataset
        self.buffer = {"data":[], "labels":[]}
        self.idx = 0

    def add(self, rows, labels):
        pass