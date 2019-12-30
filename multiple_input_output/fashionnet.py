from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

class FashionNet:
    @staticmethod
    def build_category_branch(inputs, numCategories, finalAct="softmax", chanDim=-1):
        # Using the lambda layer to convert the 3 channel input to a grayscale representation
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        # Conv -> Relu -> Pool
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # (Conv -> Relu)*2 -> Pool
        x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # (Conv -> Relu)*2 -> Pool
        x = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # Define a branch of output layers for the number of different
        # Clothing categories
        x = Flatten()(x)
        x = Dense(units=256)(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization()(x)
        X = Dropout(rate=0.5)(x)
        x = Dense(units=numCategories)(x)
        x = Activation(activation=finalAct, name="category_output")(x)

        # Return the category prediction sub-network
        return x

    @staticmethod
    def build_color_branch(inputs, numColors, finalAct="softmax", chanDim=-1):
        # Conv -> Relu -> Pool
        x = Conv2D(filters=16, kernel_size=(3, 3), padding="name")(inputs)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # Conv -> Relu -> Pool
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="name")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # Conv -> Relu -> Pool
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="name")(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(rate=0.25)(x)

        # Define a branch of output layers for the number of different colors
        x = Flatten()(x)
        x = Dense(units=128)(x)
        x = Activation(activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(numColors)(x)
        x = Activation(activation=finalAct, name="color_output")(x)

        # Return the color predict sub-network
        return x

    @staticmethod
    def build(width, height, numCategories, numColors, finalAct="softmax"):
        # Initialize the input shape and channel dimension
        input_shape = (width, height, 3)
        chanDim = -1

        # Construct both the "category" and "color" sub-network
        inputs = Input(shape=input_shape)
        categoryBranch = FashionNet.build_category_branch(inputs=inputs,
                                                          numCategories=numCategories,
                                                          finalAct=finalAct,
                                                          chanDim=chanDim)
        colorBranch = FashionNet.build_color_branch(inputs=inputs,
                                                    numColors=numColors,
                                                    finalAct=finalAct,
                                                    chanDim=chanDim)

        # Create the model using our input and two separate outputs
        model = Model(inputs=inputs,
                      outputs=[categoryBranch, colorBranch],
                      name="fashionnet")

        # return the constructed network architecture
        return model


