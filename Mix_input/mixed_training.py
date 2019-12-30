import dataset
import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
       help="Path to input dataset of house images")
args = vars(ap.parse_args())

# Construct the path to the input .txt file that caontains information
# on each house in the dataset and then load the dataset
print("[INFO] Loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = dataset.load_house_attribute(inputPath)

# Loading the house images and then scale the pixel intensities to the range [0, 1]
print("[INFO] Loading house images...")
images = dataset.load_house_images(df, args["dataset"])
images = images / 255.0

# Partition the data into training and testing splits using 75% of the data
# for training and the remaining 25% for testing
print("[INFO] Processing data...")
(trainAttrX, testAttrX, trainImagesX, testImagesX) = train_test_split(df, images, test_size=0.25, random_state=42)

# Find the largest house price in the training set and use it to scale our house prices
# to the range [0, 1] (which will lead to better training and convergence)
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"]/maxPrice
testY = testAttrX["price"]/maxPrice

# Process the house attributes data by performing min-max scaling
# on continunous features, one-shot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = dataset.process_house_attributes(df, trainAttrX, testAttrX)



# Create the MLP and CNN models
mlp = models.create_mlp(dim=trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(width=64, height=64, channel=3, regress=False)

# Create the input to our final set of layers as the "output" of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# Final FC layer head will have two dense layers, the final one is our regression head
x = Dense(units=4, activation="relu")(combinedInput)
x = Dense(units=1, activation="linear")(x)

# Final model
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# Complie the model using MAPE as loss function
optimizer = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=optimizer)

# Training the model
print("[INFO] Training model...")
model.fit([trainAttrX, trainImagesX], trainY,
          validation_data=([testAttrX, testImagesX], testY),
          epochs=200,
          batch_size=8)

# Making predictions on the testing dataset
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

# Compute the differene between the "predicted" house prices and the "actual" house
# prices
diff = preds.flatten() - testY
percentDiff = (diff/testY) * 100
absPercentDiff = np.abs(percentDiff)

# Compute the mean and standard deviation of the absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	    locale.currency(df["price"].mean(), grouping=True),
	    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))