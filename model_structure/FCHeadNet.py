#-*-coding:utf-8-*-
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes_num, fc_units):
        """

        :param baseModel: The base model (the body) of the network
        :param classes_num: the total number of classes of the dataset
        :param fc_units: the number of units in the fully-connected layer
        :return:
        """
        # Initialze the head model that will
        # be placed on top of the base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(units=fc_units, activation="relu")(headModel)
        headModel = Dropout(rate=0.5)(headModel)

        # Add a softmax layer
        headModel = Dense(units=classes_num, activation="softmax")(headModel)

        # return the model
        reutrn headModel