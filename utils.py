import json
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


def crop(image, width=32):
    origin_size = image.shape[0]
    new = image[(origin_size - width) // 2: (origin_size + width) // 2, (origin_size - width) // 2: (origin_size + width) // 2, :]
    return new


def save_trial_configs(trial_name, model_configs, train_configs, config_path='weights/'):
    configs = {
        'model': model_configs,
        'train': train_configs
    }
    with open(os.path.join(config_path, trial_name+'_config.json'), 'w') as fp:
        json.dump(configs, fp, indent=2)


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
