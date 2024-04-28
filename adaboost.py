import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import utils


class AdaBoost:
    def __init__(self, data, weak_learners, T):
        self.m = len(data['train_samples'])          # train sample size
        self.train_samples = data['train_samples']   # mx28x28
        self.train_labels = data['train_labels']     # mx1
        self.test_samples = data['test_samples']     # nx28x28
        self.test_labels = data['test_labels']       # nx1
        self.H = weak_learners
        self.k = len(weak_learners)                  # number of weak learners
        self.T = T
        self.p = 1/self.m * np.ones((1, self.m))
        self.alpha = np.zeros(self.T)
        self.ht = []
        self.predictions = np.zeros((self.T, self.k, self.m))  # foreach iteration X weak learner X training example
        self.train_loss = []
        self.test_loss = []

    def loss(self, ):
        pass

    def train(self):
        for t in range(self.T):
            incorrect_predictions = np.zeros_like(self.predictions[t])  # kxm: 1 iff weak_learner has missclassified, otherwise 0
            for j, h in enumerate(self.H):
                self.predictions[t, j, :] = h(self.train_samples)
                incorrect_predictions[j, np.where(self.predictions[t, j, :] != self.train_labels)] = 1
            eps = incorrect_predictions * np.tile(self.p, (np.shape(incorrect_predictions)[0], 1))  # kxm : error for each classifier and example
            eps = np.sum(eps, axis=1)
            j_min = np.argmin(eps)  # idx of the best h in iteration t (in terms of minimizing eps)
            self.ht.append(self.H[j_min])
            self.alpha[t] = 0.5 * np.log((1-eps[j_min]) / eps[j_min])  # scalar
            self.p *= np.exp(-1. * self.alpha[t] * self.predictions[t, j_min, :] * self.train_labels.T)  # 1xm
            self.p /= np.sum(self.p)  # keep normalization as probability distribution
            self.test(t)

    def test(self, t):
        train_predictions = np.zeros_like(self.train_labels)
        test_predictions = np.zeros_like(self.test_labels)
        for s in range(t):
            train_predictions += self.alpha[s] * self.ht[s](self.train_samples)  # 1xm
            test_predictions += self.alpha[s] * self.ht[s](self.test_samples)  # 1xn
        train_predictions = np.where(train_predictions >= 0, 1, -1)
        train_incorrect = np.zeros_like(train_predictions)
        train_incorrect[np.where(train_predictions != self.train_labels)] = 1
        test_predictions = np.where(test_predictions >= 0, 1, -1)
        test_incorrect = np.zeros_like(test_predictions)
        test_incorrect[np.where(test_predictions != self.test_labels)] = 1
        self.train_loss.append(np.sum(train_incorrect) / len(train_incorrect))
        self.test_loss.append(np.sum(test_incorrect) / len(test_incorrect))

    def get_losses(self):
        return self.train_loss, self.test_loss

