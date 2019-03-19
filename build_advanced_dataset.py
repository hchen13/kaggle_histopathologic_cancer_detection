# import the necessary packages
from imutils import paths
import pandas as pd
import random
import shutil
import os

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/cancer"
else:
    IMAGE_ROOT = "/Users/ethan/datasets/kaggle_pathology"

# set up variables
VAL_THRESH = 10000
TRAIN_INPUT_PATH = os.path.join(IMAGE_ROOT, 'train')
TRAIN_OUTPUT_PATH = os.path.join(IMAGE_ROOT, 'training')
VAL_OUTPUT_PATH = os.path.join(IMAGE_ROOT, 'validation')
valPaths = []

# load the wsi matching
wsi_df = pd.read_csv("patch_id_wsi.csv")
dataset_df = pd.read_csv(os.path.join(IMAGE_ROOT, 'train_labels.csv'))
labels = pd.read_csv(os.path.join(IMAGE_ROOT, 'train_labels.csv'), index_col=0)

# add a column with cancer presence
wsi_df["cancer"] = [ 0 if wsi.split("_")[2] == "normal" else 1 for wsi in wsi_df["wsi"]] 

# select only positive wsi and remove duplicates
positive_wsi_list = sorted(set(wsi_df[wsi_df["cancer"] == 1]['wsi']))

# shuffle the positive wsi list
random.seed(42)
random.shuffle(positive_wsi_list)

# select images from positive wsi until we have our validation dataset
while len(valPaths) <= VAL_THRESH:
    # select the images names of the first wsi name in the list and add them to
    # our validation paths
    imageNames = list(set(wsi_df[wsi_df["wsi"] == positive_wsi_list[0]]['id']))
    imageNames = [TRAIN_INPUT_PATH + '/' + p + ".tif" for p in imageNames]
    valPaths += imageNames
    positive_wsi_list.pop(0)

# extract the image names in a list
imageNames = dataset_df["id"].tolist()

# add the path to the image list and remove the paths already in the val data
trainPaths = [TRAIN_INPUT_PATH + '/' + p + ".tif" for p in imageNames]
trainPaths = list(set(trainPaths) - set(valPaths))

# define the dataset that we will be building
datasets = [
    ("training", trainPaths, TRAIN_OUTPUT_PATH),
    ("validation", valPaths, VAL_OUTPUT_PATH)
]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    # calculate basic info about the dataset
    count = 0
    totalDataset = len(imagePaths)

    # show which data split we are creating
    print("[INFO] building {} split...".format(dType))

    # if the output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] creating {} directory...".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for inputPath in imagePaths:
        # extract the filename of the input image and match
        # with the df column
        filename = inputPath.split(os.path.sep)[-1]
        name = filename[:-4]
        label = str(labels.loc[name].label)

        # build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])

        # if the label output directory does not exists, create it
        if not os.path.exists(labelPath):
            print("[INFO] creating {} directory...".format(labelPath))
            os.makedirs(labelPath)

        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)

        count += 1

        if count % 100 == 0:
            print("[INFO] ({}/{}) images copied: {}".format(
                count, totalDataset, filename
            ))