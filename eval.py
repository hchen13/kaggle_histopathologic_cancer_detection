import argparse
import json
import os

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

from nn import Prototype
from nn.prototypes import ZPNet
from utils import crop

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/cancer"
else:
    IMAGE_ROOT = "/Users/ethan/datasets/kaggle_pathology"


def eval(engine, image_size):
    # model_file = "models/trans_nasnet+fc2+pretrain-trial_4.h5"
    print('[info] preparing testing data for predictions...')

    idg = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-6,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1 / 255.0,
        # preprocessing_function=crop,
        data_format=None
    )
    batches = idg.flow_from_directory(
        os.path.join(IMAGE_ROOT, 'test'),
        class_mode=None,
        color_mode='rgb',
        target_size=(image_size, image_size),
        shuffle=False,
    )

    print("[info] predicting...")
    preds = engine.model.predict_generator(
        batches,
        verbose=1,
    )

    ids = map(
        lambda fname: os.path.basename(fname).split('.')[0],
        batches.filenames
    )

    submission = pd.DataFrame(data={
        'id': list(ids),
        'label': np.array(preds).flatten()
    }).set_index('id')

    submission.to_csv('submission.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_file', required=True, help='path to model')
    args = vars(ap.parse_args())

    folder = args['model_file'].split('/')[0]
    basename = args['model_file'].split('/')[1].split(".")[-2]

    cfg_file = "{}_config.json".format(basename)
    configs = json.load(open(os.path.join(folder, cfg_file)))
    image_shape = configs['model']['image_shape']
    image_size = image_shape[0]
    if folder == 'weights':
        configs['image_size'] = image_size
        configs['model'].pop('image_shape', None)
        engine = ZPNet(**configs['model'])
        engine.create_model()
        engine.load_weights(args['model_file'])
    else:
        image_shape = configs['model']['image_shape']
        engine = Prototype(image_size)
        engine.load(args['model_file'])

    eval(engine, image_size)
