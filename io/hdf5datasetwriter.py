import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", buffSize=1000):
        # Check to see if the output path exists, and if so, raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already exists and connot be overwritten.")

        # Open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labesl = self.db.create_dataset("labels", (dims[0], ), dtype="int")

        # Store the buffer size, then initialize the buffer itself along with the index
        # into the datasets
        self.bufSize = buffSize
        self.buffer = {"data":[], "label":[]}
        self.idx = 0

    def add(self, rows, labels):
        # Add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # Check to see if the buffer needs to be flushed to disk
        if len(self.buffer["dataa"]) > self.bufSize:
            self.flush()

    def flush(self):
        # Write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labesl[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data":[], "label":[]}


    def storeClassLabels(self, classLabels):
        # Create a dataset to store the actual class label names, then store the class labls
        dt = h5py.special_dtype(vlen=unicode)
        labelSet = self.db.create_dataset("label_name", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # Check to see if there are any other enteries in the buffer that need
        # to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # Close the dataset
        self.db.close()


