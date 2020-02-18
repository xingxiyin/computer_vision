from keras .callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # Store the output path for the figure, he path to the JSON serialied file,
        # and the starting epoch
        """

        :param figPath: The path to the output that we can use to visualize loss and accuracy over time
        :param jsonPath: An optional path used to serialize the loss and accuracy values as a JSON file
        :param startAt: The Starting epoch that training is resumed at when using ctrl+c training
        """
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt


    def on_train_begin(self, logs={}):
        # Initialize the history  dictionary fo the losses
        self.hisotry = {}

        # If the JSON hisotry path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.hisotry = json.load(open(self.jsonPath).read())

                # Check to see if a starting epoch was supplied
                if self.startAt >0:
                    # Loop over the entries in the history log and
                    # trim any entries that are past the starting epoch
                    for key in self.hisotry.keys():
                        self.hisotry[key] = self.hisotry[key][:self.startAt]


    def on_epoch_end(self, epoch, logs={}):
        # Loop over the logs and update the loss, accuracy,etc
        # for the entire training process
        for (key, value) in logs.items():
            log = self.hisotry.get(key, [])
            log.append(value)
            self.hisotry[key] = log

        # Check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            file = open(self.jsonPath, "w")
            file.write(json.dumps(self.history))
            file.close()

        # Ensure at least 2 epochs have passed before plotting
        if len(self.hisotry["loss"]) > 1:
            # Plot the training loss and accuracy
            N = np.arange(0, len(self.hisotry["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.hisotry["loss"], label="train_loss")
            plt.plot(N, self.hisotry["val_loss"], label="val_loss")
            plt.plot(N, self.hisotry["acc"], label="train_acc")
            plt.plot(N, self.hisotry["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.hisotry["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # Save the figure
            plt.savefig(self.figPath)
            plt.close()
