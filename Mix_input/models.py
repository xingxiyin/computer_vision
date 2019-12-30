from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_mlp(dim, regress=False):
    # Define MLP network for categorical/numerical inputs
    model = Sequential()
    model.add(Dense(units=8, input_dim=dim, activation="relu"))
    model.add(Dense(units=4, activation="relu"))

    # Check to see if the regression node should be added
    if regress:
        model.add(Dense(units=1, activation="linear"))

    # return the model
    return model

def create_cnn(width, height, channel, filters=(16, 32, 64), regress=False):
    # Initialize the input shape and channel dimension
    inputShape = (width, height, channel)
    chanDim = -1

    # Define the model input
    inputs = Input(shape=inputShape)

    # Loop over the number of filters
    for (index, filter) in enumerate(filters):
        # If the index is 0, then set the input appropriately
        if index == 0:
            x = inputs

        # Conv -> Relu -> BN -> POOL
        x = Conv2D(filters=filter, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


    # Flatten the volume, Then FC -> BN -> Dropout
    x = Flatten()(x)
    x = Dense(units=16)(x)
    x = Activation(activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(rate=0.5)(x)

    # Apply another FC layer, this one to match the number of nodes coming out of the MLP
    x = Dense(units=4)(x)
    x = Activation(activation="relu")(x)

    # Check to see if the regression node should be added
    if regress:
        x = Dense(units=1, activation="linear")(x)

    # Construct the CNN
    model = Model(inputs, x)

    # Return the model
    return model



