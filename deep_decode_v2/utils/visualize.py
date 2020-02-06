import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
#from sklearn.metrics import confusion_matrix
import itertools


def plot_trial_pred(prediction):
    """
    Plot predictions over time for one trial.

    :param prediction: (2d numpy array) prediction of format [example, odor]
    :return: None
    """
    fig = plt.figure(figsize=(15, 2), dpi=200)
    for j in range(5):
        plt.subplot(int('15' + str(j + 1)))
        plt.ylim(0, 1)
        plt.plot(prediction[:, j])
    plt.show()


def plot_by_odor(odor_index, all_pred, target):
    """
    Plot predictions for all trials of the same odor.

    :param odor_index: (int) odor by which to group
    :param all_pred: (3d numpy array) prediction of format [trial, time, odor]
    :param target: (2d numpy array) target of format [trial, odor]
    :return: None
    """
    odor_names = ['A', 'B', 'C', 'D', 'E']
    pred_by_odor = all_pred[(target[:, odor_index] == 1), :, :]
    n_trial = pred_by_odor.shape[0]
    fig = plt.figure(figsize=(20, 3), dpi=200)
    for j in range(5):
        ax = plt.subplot(int('15' + str(j + 1)))
        plt.xticks([9.5, 19.5, 29.5], ['-1s', '0s', '1s'])
        plt.ylim(0, 1)
        rect1 = patches.Rectangle((21, 0), 2.5, 1, linewidth=1, edgecolor='none', facecolor='red', alpha=0.2)
        ax.add_patch(rect1)
        ax.text(5, 0.75, r'$P({})$'.format(odor_names[j]), fontsize=14)
        for i in range(n_trial):
            plt.plot(pred_by_odor[i, :, j], color='blue', alpha=0.1)
    plt.suptitle('True odor ' + odor_names[odor_index], fontsize=14, fontweight='bold')
    plt.show()


def plot_latent_boundary(xx, yy, Z):
    """
    Plot decision boundary in the latent space.
    """
    cmap = ListedColormap(['deepskyblue', 'tan', 'mediumseagreen', 'purple'])
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.yticks([], [])
    plt.xticks([], [])
    for k, t in enumerate(['A', 'B', 'C', 'D']):
        xc = np.mean(xx[Z == k])
        yc = np.mean(yy[Z == k])
        plt.text(xc, yc, t, fontsize=12)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
