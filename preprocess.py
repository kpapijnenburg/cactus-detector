import os
import shutil
from distutils.dir_util import copy_tree
import pandas as pd

base = 'data'
dirs = ['train']
classes = ['has_cactus', 'no_cactus']
copy_train = False

# Initialize folders if they don't already exist
for dir_ in dirs:
    for class_ in classes:
        path = f'{base}/{dir_}/{class_}'
        if not os.path.exists(path):
            os.makedirs(path)


# Move files to their correpsonding folders
origin = 'raw-data/train'

if copy_train:
    for index, row in pd.read_csv('raw-data/train.csv').iterrows():
        if row.has_cactus == 1:
            # Move file to has_cactus folder
            shutil.copy(f'{origin}/{row.id}', f'{base}/train/has_cactus')
        else:
            # Move file to no_cactus
            shutil.copy(f'{origin}/{row.id}', f'{base}/train/no_cactus')

# Verify (values taken from the EDA)
nr_of_has_cactus = 13136
nr_of_has_no_cactus = 4364

print(f"Nr of has cactus corresponds: {nr_of_has_cactus == len(os.listdir('data/train/has_cactus'))}")
print(f"Nr of has no cactus corresponds: {nr_of_has_no_cactus == len(os.listdir('data/train/no_cactus'))}")

