from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be channels last
        # and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1   # Use for the BN, which need to know which axis to normalize over

        # If we are using "channels first", update the input shape and channels
        # dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=inputShape))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        # Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))

        # Softmax classifier
        model.add(Dense(units=classes))
        model.add(Activation(activation="softmax"))

        # return he constructed network architeture
        return model