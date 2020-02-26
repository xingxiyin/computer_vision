#-*-coding:utf-8-*-
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

class SqueezeNet:
    @staticmethod
    def squeeze(x, numFilter):
        """

        :param x: The input to the squeeze part
        :param numFilter: The number of filters that will be learned by the 1x1 CONV in the squeeze.
        :param reg: The regularization strength to 1x1 CONV layers in the residual module
        :return:
        """

        # The first part of the FIRE module consist of a number of 1x1 filter
        # squeezes on the input data followed by an activation
        # conv = Conv2D(filters=numFilter, kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(x)
        conv = Conv2D(filters=numFilter, kernel_size=(1, 1), strides=(1, 1))(x)
        act = Activation(activation="relu")(conv)

        # Return the activation result
        return act

    @staticmethod
    def expand(x, numFilter, chanDim):
        """

        :param x: The input to the squeeze part
        :param numFilter: The number of filters that will be learned by the 1x1 and 3x3 CONV in the expand.
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param reg: The regularization strength to 1x1 CONV layers in the residual module
        :return:
        """

        # Construct the 1x1 expand
        # expend_1x1 = Conv2D(filters=numFilter, kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(x)
        expend_1x1 = Conv2D(filters=numFilter, kernel_size=(1, 1), strides=(1, 1))(x)
        act_1x1 = Activation(activation="relu")(expend_1x1)

        # Construct the 3x3 expand
        # expend_3x3 = Conv2D(filters=numFilter, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False, kernel_regularizer=l2(reg))(x)
        expend_3x3 = Conv2D(filters=numFilter, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        act_3x3 = Activation(activation="relu")(expend_3x3)

        # Concatenate the 1x1 expand and 3x3 expand
        output = concatenate([act_1x1, act_3x3], axis=chanDim)

        # Return the result
        return output


    @staticmethod
    def fire(x, numSqueezeFilter, numExpandFilter, chanDim):
        """

        :param x: The input to the squeeze part
        :param numSqueezeFilter: The number of filters that will be learned by the 1x1 CONV in the squeeze.
        :param numExpandFilter: The number of filters that will be learned by the 1x1 and 3x3 CONV in the expand.
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param reg: The regularization strength to 1x1 CONV layers in the residual module
        :return:
        """
        # Construct the 1x1 squeeze
        # squeeze = SqueezeNet.squeeze(x=x, numFilter=numSqueezeFilter, reg=reg)
        squeeze = SqueezeNet.squeeze(x=x, numFilter=numSqueezeFilter)

        # Construct the expand
        # output = SqueezeNet.expand(x=squeeze, numFilter=numExpandFilter, chanDim=chanDim, reg=reg)
        output = SqueezeNet.expand(x=squeeze, numFilter=numExpandFilter, chanDim=chanDim)

        # Return the output of the FIRE module
        return output

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

        # Define the model input
        inputs = Input(shape=inputshape)

        # Block #1: CONV -> RELU -> POOL
        x = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2))(inputs)
        x = Activation(activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Block #2-4: (FIRE * 3) -> POOL
        x = SqueezeNet.fire(x=x, numSqueezeFilter=16, numExpandFilter=64)
        x = SqueezeNet.fire(x=x, numSqueezeFilter=16, numExpandFilter=64)
        x = SqueezeNet.fire(x=x, numSqueezeFilter=32, numExpandFilter=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Block #5-6: (FIRE * 4) -> POOL
        x = SqueezeNet.fire(x=x, numSqueezeFilter=32, numExpandFilter=128)
        x = SqueezeNet.fire(x=x, numSqueezeFilter=48, numExpandFilter=192)
        x = SqueezeNet.fire(x=x, numSqueezeFilter=48, numExpandFilter=192)
        x = SqueezeNet.fire(x=x, numSqueezeFilter=64, numExpandFilter=256)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Block #9-10: FIRE -> DROPOUT -> CONV -> RELU -> POOL
        x = SqueezeNet.fire(x=x, numSqueezeFilter=64, numExpandFilter=256)
        x = Dropout(rate=0.5)(x)

        x = Conv2D(filters=classes, kernel_size=(1, 1), strides=(1, 1))(x)
        x = Activation(activation="relu")(x)
        x = AveragePooling2D(pool_size=(13, 13))(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Activation(activation="softmax")(x)

        # Create the model
        model = Model(inputs, x, name="MiniGoogleNet")

        # return the constructed network archetecture
        return model