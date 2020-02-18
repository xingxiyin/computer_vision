#-*-coding:utf-8-*-
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        """

        :param width: The width of the image
        :param height: The height of the image
        :param depth: The channels of the image
        :param classes: The total number of class labels in the dataset
        :param reg: The control amount of L2 regularization which will be applying to th network
        :return:
        """
        # Initialize the model along with the input shape to be "channels last" and
        # the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block #1: First CONV -> RELU -> POOL layer set
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=inputShape, padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(rate=0.25))


        # Block #2: Second CONV -> RELU -> POOL layer set
        model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(rate=0.25))


        # Block #3: CONV -> RELU -> CONV -> RELU -> CONV -> RELU
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(rate=0.25))


        # Block #4: First set of FC -> RELU layers
        model.add(Flatten())
        model.add(Dense(units=4096, kernel_regularizer=l2(reg)))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))


        # Block #5: Second set of FC -> RELU layers
        model.add(Dense(units=1000, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))

        # Softmax classifier
        model.add(Dense(units=classes, kernel_regularizer=l2(reg)))
        model.add(Activation(activation="softmax"))

        # return the constructed network architecture
        return model



