#-*-coding:utf-8-*-
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K

class MiniGooogleNet:
    @staticmethod
    def conv_model(x, filters, kernel_size, stride, chanDim=-1, padding="same"):
        """

        :param x: The input layer to the function
        :param filters: The number of filters the CONV layer is going to learn
        :param kernel_size: The size of each of the filter that will be learned
        :param stride: THe stride of the CONV layer
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param padding: The type of padding to be applied to the CONV layer
        :return:
        """
        # Define a CON -> BN -> RELU pattern
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation(activation="relu")(x)

        # return the result
        return x


    @staticmethod
    def inception_module(x, branch1x1, branch3x3, chanDim=-1):
        """

        :param x: The input layer to the function
        :param branch1x1: The number of filters of inception branch #1: 1*1 CON
        :param branch3x3: The number of filters of inception branch #2: 3*3 CON
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :return:
        """
        # Define two
        conv_1x1 = MiniGooogleNet.conv_model(x=x, filters=branch1x1, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim)
        conv_3x3 = MiniGooogleNet.conv_model(x=x, filters=branch3x3, kernel_size=(3, 3), stride=(1, 1), chanDim=chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

        # return the block
        return x

    @staticmethod
    def downsample_module(x, filters, chanDim=-1):
        """

        :param x: The input layer to the function
        :param filters: The number of the filters to the conv layer
        :param chanDim:  The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :return:
        """
        # Define the CONV module and POOL, then concatenate across the chanel dimensions
        conv_3x3 = MiniGooogleNet.conv_model(x, filters=filters, kernel_size=(3, 3), stride=(2, 2), chanDim=chanDim, padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)

        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes):
        """

        :param width: The width of the input
        :param height: The height of the input
        :param depth: The depth of the input
        :param classes: The number of classes
        :return:
        """
        # Initialize the input shape to be "channels last" and the channels dimension itself
        inputshape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputshape = (depth, height, width)
            chanDim = 1

        # Define the model input and first CONV module
        inputs = Input(shape=inputshape)
        x = MiniGooogleNet.conv_model(x=inputs, filters=96, kernel_size=(3, 3), stride=(1, 1), chanDim=chanDim)

        # Two Inception modules followed by a downsample module
        x = MiniGooogleNet.inception_module(x=x, branch1x1=32, branch3x3=32, chanDim=chanDim)
        x = MiniGooogleNet.inception_module(x=x, branch1x1=32, branch3x3=48, chanDim=chanDim)
        x = MiniGooogleNet.downsample_module(x=x, filters=80, chanDim=chanDim)

        # four Inception modules followed by a downsample module
        x = MiniGooogleNet.inception_module(x=x, branch1x1=112, branch3x3=48, chanDim=chanDim)
        x = MiniGooogleNet.inception_module(x=x, branch1x1=96, branch3x3=64, chanDim=chanDim)
        x = MiniGooogleNet.inception_module(x=x, branch1x1=80, branch3x3=80, chanDim=chanDim)
        x = MiniGooogleNet.inception_module(x=x, branch1x1=48, branch3x3=96, chanDim=chanDim)
        x = MiniGooogleNet.downsample_module(x=x, filters=96, chanDim=chanDim)

        # Two Inception modules followed by global POOL and dropout
        x = MiniGooogleNet.inception_module(x=x, branch1x1=176, branch3x3=160, chanDim=chanDim)
        x = MiniGooogleNet.inception_module(x=x, branch1x1=176, branch3x3=160, chanDim=chanDim)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(rate=0.5)(x)

        # Sfotmax classifier
        x = Flatten()(x)
        x = Dense(units=classes)(x)
        x = Activation(activation="softmax")(x)

        # Create the model
        model = Model(inputs, x, name="MiniGoogleNet")

        # return the constructed network archetecture
        return model