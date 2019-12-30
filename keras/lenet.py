from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, channel, num_classes, chanDim=-1):
        # Initialize the model
        model = Sequential()

        # Network structure CONV -> RELU -> POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(width, height, channel)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Network structure CONV -> RELU -> POOL
        model.add(Conv2D(59, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten Layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Define the FC layer
        model.add(Dense(num_classes))

        # Softmax classifier
        model.add(Activation("softmax"))

        # Reture the model
        return model
