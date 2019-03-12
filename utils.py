import os
import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def display_image(*images, title='image display', col=None):
    from matplotlib import pyplot as plt
    if col is None:
        col = len(images)
    plt.figure(figsize=(16, 9))
    plt.title(title)
    row = np.math.ceil(len(images) / col)
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()


class Chart(TensorBoard):

    def __init__(self, *args, **kwargs):
        super(Chart, self).__init__(*args, **kwargs)

    def draw(self, logs, num):
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, num)
            self.writer.flush()

    # def on_batch_end(self, num, logs=None):
    #     loss = logs['loss']
    #     acc = logs['acc']
    #     self.write_log(['loss', 'acc'], [loss, acc], num)
    #
    # def on_epoch_end(self, epoch, logs=None):
    #     loss = logs['val_loss']
    #     acc = logs['val_acc']
    #     self.write_log(['loss', 'acc'], [loss, acc], epoch)
