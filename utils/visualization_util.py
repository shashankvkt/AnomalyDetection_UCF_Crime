import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import configuration as cfg
import cv2
from matplotlib.animation import FuncAnimation


def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
    num_frames = len(clip)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        if convert_bgr:
            frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        else:
            frame = clip[i]
        plt.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)
    if save_gif:
        anim.save(file_path, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()


def visualize_confusion_matrix(data, x_labels, y_labels, f_name):

    assert data.shape[0] == y_labels
    assert data.shape[1] == x_labels

    plt.figure(figsize=(5,5))
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.YlOrBr)
    plt.xticks(np.arange(4), x_labels, fontsize=12, )
    plt.yticks(np.arange(4), y_labels, fontsize=12)

    fmt = '.2f'
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, format(data[i, j], fmt), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(f_name)


