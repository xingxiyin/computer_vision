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
from keras.regularizers import l2
from keras import backend as K

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, filters, kernel_size, stride, chanDim, padding="same", reg=0.005, name=None):
        """

        :param x: The input to the network
        :param filters: The number of filters the convolutional layer will learn
        :param kernel_size: The filter size for the convolutional layer
        :param stride: The stride for the convolution
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param padding: The padding of the convolution layer
        :param reg: The L2 weight decay strength
        :param name: The name of the conv_module
        :return:
        """
        # Initialize the CONV, BN, RELU layer names
        (convName, bnName, actName) = (None, None, None)

        # If a layer name was supplied, prepend it
        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"

        # Define a CONV -> BN -> RELU pattern
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation(activation="relu", name=actName)(x)

        # Return the block
        return x


    @staticmethod
    def inception_module(x, conv1x1, conv3x3Reduce, conv3x3, conv5x5Reduce, conv5x5, conv1x1Proj, chanDim=-1, name=None, reg=0.0005):
        """
        The Iception module includes for branches, the outputs of then are concatenated along the channel dimension
        :param x: The input to the network
        :param conv1x1: The number of filter of the first branch: 1x1 convolution
        :param conv3x3Reduce: The number of filter of the second branch: 1x1 convolution
        :param conv3x3: The number of filter of the second branch: 3x3 convolution
        :param conv5x5Reduce: The number of filter of the third branch: 1x1 convolution
        :param conv5x5: The number of filter of the third branch: 5x5 convolution
        :param conv1x1Proj: The number of filter of the fourth branch: 1x1 convolution
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param name:The name of the conv_module
        :return:
        """
        # Define the first branch of the Inception module which consists of 1x1 convolution
        first = DeeperGoogleNet.conv_module(x=x, filters=conv1x1, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim, reg=reg, name=name+"_first")

        # Define the second branch of the Inception module which consists of 1x1 and 3x3 convolutions
        second = DeeperGoogleNet.conv_module(x=x, filters=conv3x3Reduce, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim, reg=reg, name=name + "_second1")
        second = DeeperGoogleNet.conv_module(x=second, filters=conv3x3, kernel_size=(3, 3), stride=(1, 1), chanDim=chanDim, reg=reg, name=name + "_second2")

        # Define the third branch of the Inception module which consists 1x1 and 5x5 convolutions
        third = DeeperGoogleNet.conv_module(x=x, filters=conv5x5Reduce, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim, reg=reg, name=name + "_third1")
        third = DeeperGoogleNet.conv_module(x=third, filters=conv5x5, kernel_size=(5, 5), stride=(1, 1), chanDim=chanDim, reg=reg, name=name + "_third2")

        # Define the fourth branch of the Inception module which si the POOL projection
        fourth = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name=name + "_pool")(x)
        fourth = DeeperGoogleNet.conv_module(x=fourth, filters=conv1x1Proj, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim, reg=reg, name=name + "_fourth")

        # Concatenate across the channel dimension
        x = concatenate([first, second, third, fourth], axis=chanDim, name=name + "_mixed")

        # Return the block
        return x


    @staticmethod
    def build(width, height, depth, classes, reg=0.005):
        """

        :param width: The width of the input image
        :param height: The height of the input image
        :param depth: The depth of the input image
        :param classes:The total number of class
        :param reg: The regulazrization term for L2 weight decay
        :return:
        """
        # Initialize the input shape to be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim

        # Define the model input, followed by a sequence of CONV -> POOL -> (CONV *2) -> POOL layers
        inputs = Input(shape=inputShape)
        x = DeeperGoogleNet.conv_module(x=inputs, filters=64, kernel_size=(5, 5), stride=(1, 1), chanDim=chanDim, reg=reg. name="block1")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogleNet.conv_module(x=x, filters=64, kernel_size=(1, 1), stride=(1, 1), chanDim=chanDim, reg=reg, name="block2")
        x = DeeperGoogleNet.conv_module(x=x, filters=192, kernel_size=(3, 3), stride=(1, 1), chanDim=chanDim, reg=reg,name="block23")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool2 ")(x)

        # Apply two Inception modules followed by a POOL
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=64, conv3x3Reduce=96, conv3x3=128, conv5x5Reduce=16,
                                             conv5x5=32, conv1x1Proj=32, chanDim=chanDim, name="Incept1", reg=reg)
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=128, conv3x3Reduce=128, conv3x3=192, conv5x5Reduce=32,
                                             conv5x5=96, conv1x1Proj=64, chanDim=chanDim, name="Incept2", reg=reg)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        # Apply five Inception modules followed by POOL
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=192, conv3x3Reduce=96, conv3x3=208, conv5x5Reduce=16,
                                             conv5x5=48, conv1x1Proj=64, chanDim=chanDim, name="Incept4", reg=reg)
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=160, conv3x3Reduce=112, conv3x3=224, conv5x5Reduce=24,
                                             conv5x5=64, conv1x1Proj=64, chanDim=chanDim, name="Incept5", reg=reg)
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=128, conv3x3Reduce=128, conv3x3=256, conv5x5Reduce=24,
                                             conv5x5=64, conv1x1Proj=64, chanDim=chanDim, name="Incept6", reg=reg)
        x = DeeperGoogleNet.inception_module(x=x, conv1x1=256, conv3x3Reduce=160, conv3x3=320, conv5x5Reduce=32,
                                             conv5x5=128, conv1x1Proj=128, chanDim=chanDim, name="Incept7", reg=reg)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool4")(x)  # Volume size is 4x4xclasses

        # Apply a POOL layer (average) followed by dropout
        x = AveragePooling2D(pool_size=(4, 4), name="pool5")(x)
        x = Dropout(rate=0.4, name="drop0")(x)

        # Softmax classifier
        x = Flatten(name="Flatten")(x)
        x = Dense(units=classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation(activation="softmax", name="softmax")(x)

        # Create the model
        model = Model(inputs, x, name="GoogleNet")

        # Return the constructed network architecture
        return model








