#-*-coding:utf-8-*-
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(x, filters, stride, chanDim, red=False, reg=0.001, bnEps=2e-5, bnMom=0.9):
        """

        :param x: The input to the residual module
        :param filters: The number of filters that will be learned by the final CONV in the bottleneck.
                        The first tow CONV layers will learn filters/4 filters
        :param stride: The stride of the convolution
        :param chanDim: The channel dimension. indecated that either "channels last" or "channels first" ordering. default is channels last
        :param red: A boolean value which will control whether we are reducing spatial dimensions (True) or not (False)
        :param reg: The regularization strenght to all CONV layers in the residual module
        :param bnEps: Responsible for avoiding "divison by zero" errors when normalizing inputs
        :param bnMom: Control the momentum for the moving average
        :return:
        """
        # The shortcut branch of the ResNet module should be initialized as the input (identitu) data
        shortcut = x

        # The first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        act1 = Activation(activation="relu")(bn1)
        conv1 = Conv2D(filters=int(filters/4), kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # The second block of the ResNet module are 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation(activation="relu")(bn2)
        conv2 = Conv2D(filters=int(filters/4), kernel_size=(3, 3), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act2)

        # The third block of the ResNet module are the 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation(activation="relu")(bn3)
        conv3 = Conv2D(filters=int(filters/4), kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # If we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(filters=filters, kernel_regularizer=(1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # Add together the shortcut and final CONV
        x = add([conv3, shortcut])

        # Return the addition as the output of the ResNet module
        return x


    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        """

        :param width: The width of the input image
        :param height: The height of the input image
        :param depth: The depth of the input image
        :param classes: The number of classes of the images
        :param stages: The list of stages use to construct the ResNet architecture
        :param filters: The list of filter use to construct of the ResNet architecture
        :param reg: The regularization strenght to all CONV layers in the residual module
        :param bnEps: Responsible for avoiding "divison by zero" errors when normalizing inputs
        :param bnMom: Control the momentum for the moving average
        :return:
        """
        # Initialize the input shape to be "channel last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Set the input and apply batch normalization
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMon)(inputs)

        # Loop over the number of stages
        for idx in range(0, len(stages)):
            # Initialize the stride, then apply a residual module used to reduce the spatial aize of the input volume
            stride = (1, 1) if idx == 0 else (2, 2)
            x = ResNet.residual_module(x=x, filters=filters[idx], stride=stride, chanDim=chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # Loop over the number of layers in the stage
            for stage_idx in range(0, stages[idx] -1)
                # Apply a ResNet module
                x = ResNet.residual_module(x=x, filters=filters[idx], stride=(1, 1), chanDim=chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

        # Apply BN -> ACTIVATION -> POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation(activation="relu")(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Dense(units=classes, kernel_regularizer=l2(reg))(x)
        x = Activation(activation="softmax")(x)

        # Create the model
        model = Model(inputs, x, name="resnet")

        # Return the constructed network architecture
        return model































