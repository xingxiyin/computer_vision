import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Initialize the list of weights matrices, then store the network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Start looping from the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0, len(layers)-2):
            # Randomly initilize a weight matrix connecting the number of
            # number of nodes in each respective layer together, adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # The last two layers are a special case where the input connections
        # need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))


    def __repr__(self):
        # Construct and return a string that represents the network architecture
        return "NeuralNetwork:{}".format("-".join(str(layer) for layer in self.layers))


    def sigmoid(self, x):
        # Compute and return the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))


    def sigmoid_deriv(self, x):
        # Compute the derivative of the sigmoid function
        return x * (1 - x)


    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # Insert a column of 1's as the last entry in the feature
        # matrix  -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each individual data point and train our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))


    def fit_partial(self, x, y):
        # Construct our list of output activations for each layer as
        # our data point flows through the network; the first
        # activation is a special case -- it's just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FeedForward
        # Loop over the layers in the network
        for layer in np.arange(o, len(self.W)):
            # Feedforward the activation at the current layer by taking
            # the dot product between the activation and the weight matrix -- this
            # is called the "net input" to the current layer
            net = A[layer].dot(self.W[layer])

            # Computing the "net ouput" is simply applying our non-linear activation
            # function to the net input
            out = self.sigmoid(net)

            # Once we have the net output, add it to our list of activations
            A.append(out)

        # BackPropagation
        # The first phase of backprogation is to compute the difference between
        # our "prediction" (The final output activation in the activations list)
        # and the ture target value
        error = A[-1] -y

        # From here, we need to apply the chain rule and build our list of
        # deltas 'D'; the first entry in the deltas is simply the error of the
        # ouput layer times the derivative of our activations function for the output value
        deltas = [error * self.sigmoid_deriv(A[-1])]

        # Implement with a for loop -- simply loop over the layers in the reverse order
        # (ignoring the last two since we already have taken into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # The delta for the current layer is equal to the delta of the
            # previous layer dotted with the weight matrix of the current layer
            # followed by multiplying the delta by the derivative of nonlinear activation
            # function for the activation of the current layer
            delta = deltas[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            deltas.append(delta)


        # Since we looped over the layers in reverse order we need to reverse the deltas
        deltas = deltas[::-1]

        # Weight Update phase
        # Loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer activations with
            # their respective deltas, then multiplying this value by some small learning
            # rate and adding to our weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] += -self.alpha * A[layer].T.dot(deltas[layer])


    def predict(self, X, addBias=True):
        # Initialize the output prediction as the input features -- this value willb
        # be (forward) propagated through the network to obtain the final prediction
        pred = np.atleast_2d(X)

        # Check to see if the bias column should be added
        if addBias:
            # Inset a column of 1's as the last entry in the feature
            # matrix (bias)
            pred = np.c_[p, np.ones((pred.shape[0]))]

        # Loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # Computing the output prediction is as simple as taking
            # the dot product between the current activation value 'p'
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation function
            pred = self.sigmoid(np.dot(pred, self.W[layer]))

        # return the predicted value
        return pred

    def calculate_loss(self, X, targets):
        # Making predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=True)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # Return the loss
        return loss