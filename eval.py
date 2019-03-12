import glob
import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from nn import Prototype
import pandas as pd

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/cancer"
else:
    IMAGE_ROOT = "/Users/ethan/datasets/kaggle_pathology"


def test_data_generator(image_size):
    batch_size = 32
    print("[info] loading test images in batches of {}".format(batch_size))
    test_dir = os.path.join(IMAGE_ROOT, 'test', 'test')
    image_files = glob.glob(test_dir + '/*.tif')
    print("{} images found, dividing into {} batches.".format(
        len(image_files),
        int(len(image_files) / batch_size)
    ))
    for b in range(0, len(image_files), batch_size):


        if b > 3:
            break


        batch_files = image_files[b : b + batch_size]
        image_list = []
        for i, file in enumerate(batch_files):
            img = cv2.imread(file)
            if img.shape[:2] != (image_size, image_size):
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            image_list.append(img)
        batch = np.array(image_list)
        yield batch





if __name__ == '__main__':
    image_size = 96
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
        # preprocessing_function=lambda x: x / 255,
        data_format=None
    )
    batches = idg.flow_from_directory(
        os.path.join(IMAGE_ROOT, 'test'),
        class_mode=None,
        color_mode='rgb',
        target_size=(image_size, image_size),
        shuffle=False
    )

    print('[info] loading nn model...')
    model_file = "models/trained.h5"
    engine = Prototype(image_size)
    engine.load(model_file)

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