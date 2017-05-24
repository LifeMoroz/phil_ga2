import random

import numpy as np
import math

from NN1.image_loader import ImageLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Neural network for digits recognition (with back propagation)')

    parser.add_argument('--training_path', type=str, dest='training_path',
                        help='Path to training image set', default='../NN1/training_set')
    parser.add_argument('--test_path', type=str, dest='test_path',
                        help='Path to test image set', default='../NN1/test_set')
    parser.add_argument('-l', '--hidden_layers', type=int, nargs='+', dest='hidden',
                        help='List of hidden layers sizes', default=[10])
    parser.add_argument('-r', '--learning_rate', type=float, dest='learning_rate',
                        help='Learning rate', default=0.5)
    parser.add_argument('-d', '--rate_decay', type=float, dest='rate_decay',
                        help='Learning rate decay coefficient', default=0.003)
    parser.add_argument('-m', '--momentum', type=float, dest='momentum',
                        help='Training momentum (fraction of prev change that\'s added to current iteration',
                        default=0.5)
    parser.add_argument('-i', '--max_iterations', type=int, dest='max_iterations',
                        help='Maximum number of training iterations', default=100)
    parser.add_argument('-e', '--error_threshold', type=float, dest='error_threshold',
                        help='Absolute error value at which learning stops', default=0.001)

    return parser.parse_args()


class NN(object):
    def __init__(self, args: dict = None):
        if not args:
            args = vars(parse_args())

        # initialize parameters
        self.training_path = args['training_path']
        self.test_path = args['test_path']
        self.iterations = args['max_iterations']
        self.error_threshold = args['error_threshold']
        self.learning_rate = args['learning_rate']
        self.momentum = args['momentum']
        self.rate_decay = args['rate_decay']

        # initialize arrays
        self.input = 324
        self.hidden = args['hidden']
        self.output = 10

        # set up array of 1s for activations
        self.input_activations = [1.0] * self.input
        self.hidden_activations = [1.0] * self.hidden[0]
        self.output_activations = [1.0] * self.output

        # create randomized weights
        input_range = 1.0 / self.input ** (1 / 2)
        output_range = 1.0 / self.hidden[-1] ** (1 / 2)

        # initialize weights
        self.hidden_weights = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden[0]))
        self.output_weights = np.random.normal(loc=0, scale=output_range, size=(self.hidden[-1], self.output))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    @staticmethod
    def dsigmoid(y):
        return y * (1.0 - y)

    # using tanh over logistic sigmoid is recommended
    @staticmethod
    def tanh(x):
        return math.tanh(x)

    # derivative for tanh sigmoid
    @staticmethod
    def dtanh(y):
        return 1 - y * y

    def feed_forward(self, inputs):
        # input activations
        for i in range(self.input):
            self.input_activations[i] = inputs[i]

        for h in range(self.hidden[0]):
            activation = 0.0
            for i in range(self.input):
                activation += self.input_activations[i] * self.hidden_weights[i][h]
            self.hidden_activations[h] = self.tanh(activation)

        for k in range(self.output):
            activation = 0.0
            for j in range(self.hidden[-1]):
                activation += self.hidden_activations[j] * self.output_weights[j][k]
            self.output_activations[k] = self.sigmoid(activation)

        return self.output_activations[:]

    def calc_error(self, targets):
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.output_activations[k]) ** 2
        return error

    def execute(self, training_set):
        error = 0.0
        random.shuffle(training_set)
        for p in training_set:
            inputs = p[0]
            targets = p[1]
            self.feed_forward(inputs)
            error += self.calc_error(targets)
        return error
