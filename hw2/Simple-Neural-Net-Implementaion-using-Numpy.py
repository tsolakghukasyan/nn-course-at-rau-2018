# coding: utf-8

# # HW2

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# #### Implementing Simple Artificial Neural Network for multiclass classification

class SingleLayerNetwork:
    """ A single layer neural net for multiclass classification. """
    
    def __init__(self):
        self.weights = None
        self.learning_rate = None
    
    def fit(self, X, y, epoch=10, learning_rate=0.001):
        """
        Learns the net's parameters based on provided labelled data.
        
        :param X: the train objects
        :param y: the one-hot encoded classes of given objects
        :param epoch: the number of passes over the entire dataset during training
        :param learning_rate: size of gradient descent step
        """
        self.__initialize_net(X, y, learning_rate)  # randomly initialize network's weights
        X = np.insert(X, 0, 1, axis=1)              # add constant feature
        for i in range(epoch):
            _X, _y = shuffle(X, y)                  # shuffle data before each epoch
            for obj, label in zip(_X, _y):
                self.__update_weights(obj, label)   # update network's weights

    def predict(self, X):
        """
        Predicts the class for given objects.
        
        :param X: an array of objects
        :returns: predicted classes of given objects in one-hot encoding
        """
        scores = self.__predict(np.insert(X, 0, 1, axis=1))
        return self.labels[np.argmax(scores, axis=1)]
    
    def __initialize_net(self, X, y, learning_rate):
        self.labels = np.unique(y, axis=0)
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.01, 0.01, (len(X[0]) + 1, len(self.labels)))
    
    def __predict(self, X):
        return self.__softmax(X.dot(self.weights))

    def __backprop(self, X, y, scores):
        diff = scores - y
        self.weights -= self.learning_rate * np.outer(diff, X).T

    def __update_weights(self, X, y):
        scores = self.__predict(X)     # forward pass
        self.__backprop(X, y, scores)  # derivative backpropagation
        
    def __softmax(self, scores):
        e_x = np.exp(scores - np.max(scores))
        return e_x / e_x.sum(axis=0)


# #### Loading and visualizing the MNIST dataset

mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

plt.imshow(mnist.train.images[0].reshape((28, 28)))
plt.colorbar()


# #### Training the net to classify MNIST

net = SingleLayerNetwork()
net.fit(mnist.train.images, mnist.train.labels)

# compute classification accuracy on test images
predicted = net.predict(mnist.test.images)
true_labels = mnist.test.labels
print('accuracy:', np.sum(np.argmax(predicted, axis=1) == np.argmax(true_labels, axis=1)) / len(true_labels))

