import os
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


def load_data(classes: tuple, nof_samples_per_class: tuple) -> dict:
    assert len(classes) == 2  # enforce binary problem
    c1, c2 = classes
    c1_size, c2_size = nof_samples_per_class
    (X1, y1), (X2, y2) = mnist.load_data()
    (X, y) = (np.concatenate((X1, X2)), np.concatenate((y1, y2)))
    X = np.concatenate((X[(y == c1), :][:c1_size, :, :], X[(y == c2), :][:c2_size, :, :]))
    X = X.reshape(X.shape[0], -1)
    y = np.concatenate((y[(y == c1)][:c1_size], y[(y == c2)][:c2_size]))
    y = y.astype(np.float64)
    y[y == c1] = 1
    y[y == c2] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    data = dict()
    data['train_samples'] = X_train
    data['test_samples'] = X_test
    data['train_labels'] = y_train
    data['test_labels'] = y_test
    return data


def above_threshold_classifier(j, threshold):
    return lambda data: np.where(data[:, j] >= threshold, 1, -1)


def below_threshold_classifier(j, threshold):
    return lambda data: np.where(data[:, j] <= threshold, 1, -1)


def show_image(gallery, idx, fname):
    # zero_batch = gallery
    reshaped = np.asarray(np.reshape(gallery[idx, :], (28, 28)))
    plt.imshow(reshaped, cmap='gray', vmin=0, vmax=255)
    plt.savefig(fname)


def normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def show_scores(score1, score1_title, score2, score2_title, nof_features, nof_samples):
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
    manifest_plt = ax1.imshow(score1.reshape(28, 28))
    fig.colorbar(manifest_plt, ax=ax1)
    ax1.set_title(score1_title)
    relief_plt = ax2.imshow(score2.reshape(28, 28))
    fig.colorbar(relief_plt, ax=ax2)
    ax2.set_title(score2_title)
    plt.savefig(f'plots/scores_visualization_{nof_features}_features{nof_samples}_samples.png')
    plt.cla()


def plot_errors(train_error, test_error, fs_algorithm, adaboost_timesteps, nof_features, nof_samples):
    plt.figure()
    plt.plot(train_error, label='train_error')
    plt.plot(test_error, label='test_error')
    plt.grid(True)
    plt.xlabel('Iteration (T)')
    plt.ylabel('Error Percentage')
    plt.title(f'AdaBoost Error with {fs_algorithm}-Based Weak Learners\n'
              f'{nof_samples} samples, {adaboost_timesteps} iterations, {nof_features} weak_learners'
    )
    plt.legend()
    plt.savefig(f'plots/AdaBoost_{fs_algorithm}_{nof_features}_features_{nof_samples}_samples.png')
    plt.cla()

