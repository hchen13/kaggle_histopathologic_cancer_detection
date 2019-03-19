import glob
import os

import cv2
import pandas as pd

from utils import ensure_dir, display_image

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/cancer"
else:
    IMAGE_ROOT = "/Users/ethan/datasets/kaggle_pathology"

TRAIN_DIR = os.path.join(IMAGE_ROOT, 'train')
VALID_DIR = os.path.join(IMAGE_ROOT, 'valid')

def restructure_data(verbose=500):
    ensure_dir(os.path.join(TRAIN_DIR, '0'))
    ensure_dir(os.path.join(TRAIN_DIR, '1'))
    ensure_dir(os.path.join(VALID_DIR, '0'))
    ensure_dir(os.path.join(VALID_DIR, '1'))
    df = pd.read_csv(os.path.join(IMAGE_ROOT, 'train_labels.csv'), index_col=0, engine='python')
    print('[info] moving training images into labeled subfolders...')
    train_files = glob.glob(TRAIN_DIR + '/*.tif')
    for i, train_file in enumerate(train_files):
        filename = os.path.basename(train_file)
        train_id = filename.split('.')[0]
        label = df.loc[train_id].label
        dest = os.path.join(TRAIN_DIR, str(label), filename)
        os.rename(train_file, dest)
        if (i + 1) % verbose == 0:
            print("{}/{} images moved".format(i + 1, len(train_files)))

    print('[info] moving validation images into labeled subfolders...')
    valid_files = glob.glob(VALID_DIR + '/*.tif')
    for i, valid_file in enumerate(valid_files):
        filename = os.path.basename(valid_file)
        valid_id = filename.split('.')[0]
        label = df.loc[valid_id].label
        dest = os.path.join(VALID_DIR, str(label), filename)
        os.rename(valid_file, dest)
        if (i + 1) % verbose == 0:
            print("{}/{} images moved".format(i + 1, len(valid_files)))


def create_validation_set(verbose=500):
    validation_size = 10000
    label_file = os.path.join(IMAGE_ROOT, 'train_labels.csv')
    label_index = pd.read_csv(label_file, engine='python')
    from sklearn.utils import shuffle
    label_index = shuffle(label_index)
    print("[info] preparing {} validation data...".format(validation_size))

    valid_files = glob.glob(VALID_DIR + '/*/*.tif')

    validation_size = validation_size - len(valid_files)
    if validation_size <= 0:
        print("There are sufficient amount of validation files, done!\n")
        return
    ensure_dir(VALID_DIR)
    validation_set = label_index[:validation_size]
    n = validation_set.shape[0]
    print("[info] moving {} images from train folder as validation set...".format(n))
    for i in range(n):
        file_name = "{}.tif".format(validation_set.iloc[i]['id'])
        file_path = os.path.join(TRAIN_DIR, file_name)
        destination_path = os.path.join(VALID_DIR, file_name)
        os.rename(file_path, destination_path)
        if (i + 1) % verbose == 0:
            print("{}/{} files moved".format(i + 1, n))

if __name__ == '__main__':
    create_validation_set()
    restructure_data()