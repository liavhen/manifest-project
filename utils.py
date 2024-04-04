import os
import numpy as np


def load_data():
    train_data = np.loadtxt('./data/MNIST_train_images.csv', delimiter=',')
    train_labels = np.loadtxt('./data/MNIST_train_labels.csv', delimiter=',')
    test_data = np.loadtxt('./data/MNIST_test_images.csv', delimiter=',')
    test_labels = np.loadtxt('./data/MNIST_test_labels.csv', delimiter=',')
    return train_data, train_labels, test_data, test_labels

