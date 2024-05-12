# EECE 5890 Term Project
# Created by: Mina Gaffney
# Created on: 03/05/2024

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os
import warnings
from enum import Enum
from os.path import exists, splitext
import cv2
import numpy as np

# need to use a custom fxn to load video data because of custom video CODEC for OCVL data
# AOIP data could be loaded without special fxn
# Function from resources.py within F-Cell Repo by RFC
class ResourceType(Enum):
    CUSTOM = 1
    IMAGE2D = 2
    IMAGE3D = 3
    IMAGE4D = 4
    TEXT = 5
    COORDINATES = 6

class ResourceLoader:
    def __init__(self, file_paths):

        if not file_paths:
            raise FileNotFoundError("File path " + file_paths + " doesn't exist!")

        # If this isn't a list, then make it one.
        if not hasattr(file_paths, "__len__") or isinstance(file_paths, str):
            self.file_paths = np.array([file_paths])
        else:
            self.file_paths = file_paths

        for file in self.file_paths:
            if not exists(file):
                raise FileNotFoundError("File path " + file + " doesn't exist! Load failed.")

class ResourceSaver:
    def __init__(self, file_paths, dataset, metadata):

        # If this isn't a list, then make it one.
        if not hasattr(file_paths, "__len__") or isinstance(file_paths, str):
            self.file_paths = np.array([file_paths])
        else:
            self.file_paths = file_paths

        for file in self.file_paths:
            if not exists(file):
                raise FileNotFoundError("File path " + file + " doesn't exist! Load failed.")

class Resource:
    def __init__(self, dattype=ResourceType.CUSTOM, name="", data=np.empty([1]), metadict={}):
        self.type = dattype
        self.name = name
        self.data = data
        self.metadict = metadict



def load_video(video_path):
    # Load the video data.
    vid = cv2.VideoCapture(video_path)

    framerate = -1

#    warnings.warn("Videos are currently only loaded as grayscale.")
    if vid.isOpened():
        framerate = vid.get(cv2.CAP_PROP_FPS)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Grab one frame, so we can find out the datatype for our array.
        ret, frm = vid.read()

        video_data = np.empty([height, width, num_frames], dtype=frm.dtype)
        video_data[..., 0] = frm[..., 0]
    else:
        warnings.warn("Failed to open video: "+video_path)

    i = 1
    while vid.isOpened():
        ret, frm = vid.read()
        if ret:
            video_data[..., i] = frm[..., 0]
            i += 1
        else:
            break

    vid.release()

    meta = {"framerate": framerate}

    return Resource(dattype=ResourceType.IMAGE3D, name=os.path.basename(video_path), data=video_data, metadict=meta)

# User selects dataset from file - Load in observations spreadsheet
root = Tk()
fName = filedialog.askopenfilename(title="Select Brennan_AIQManuscriptDataSpreadsheet.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# Reading in dataset as dataframe
dataset=pd.read_excel(fName, sheet_name=['AOIP','OCVL'])
AOIP_df=dataset['AOIP']
OCVL_df=dataset['OCVL']

Aug_AOIP_df = AOIP_df.copy()
Aug_OCVL_df = OCVL_df.copy()
Aug_AOIP_df['Confocal Image name'] = Aug_AOIP_df['Confocal Image name'].str.replace('.tif', '_flipped.tif')
Aug_AOIP_df['Split Image name'] = Aug_AOIP_df['Split Image name'].str.replace('.tif', '_flipped.tif')
Aug_OCVL_df['Confocal Image name'] = Aug_OCVL_df['Confocal Image name'].str.replace('.tif', '_flipped.png')
Aug_OCVL_df['Split Image name'] = Aug_OCVL_df['Split Image name'].str.replace('.tif', '_flipped.png')

Comb_Aug_AOIP_df = pd.concat([AOIP_df,Aug_AOIP_df])
Comb_Aug_OCVL_df = pd.concat([OCVL_df,Aug_OCVL_df])

fName_1_split = fName.split('/')
fName_1_sliced = fName_1_split[0:9:1] #This probs shouldn't be hardcoded but alas I'm lazy
fName_1_joined = '/'.join(fName_1_sliced)
out_dir = Path(fName_1_joined)

with pd.ExcelWriter(out_dir.joinpath('Brennan_AIQManuscriptDataSpreadsheet_wAugmentedData.xlsx')) as writer:
    Comb_Aug_AOIP_df.to_excel(writer, sheet_name='AOIP')
    Comb_Aug_OCVL_df.to_excel(writer, sheet_name='OCVL')

print(' ')
print('Dataset loaded.')

# TODO: Load in folders containing the confocal and split images and videos
#  (4 separate folders based on current data organization structure)

# Note: this will be written to expect a very specific folder structure as follows:
#   Lvl 1: Training_Data
#       Lvl 2: Averaged Images
#           Lvl 3: confocal - averaged confocal tifs or pngs
#           Lvl 3: split - averaged split tifs or pngs
#       Lvl 2: RawVideos
#           Lvl 3: confocal - raw confocal avis
#           Lvl 3: split - raw split avis


pName = filedialog.askdirectory(title="Select the folder containing all raw videos and averaged images of interest.", parent=root)

if not pName:
    quit()

# defining path names according to above folder structure
searchpath = Path(pName)
avgimgpath = Path.joinpath(searchpath, "AveragedImages")
avgimg_subdirs = [x for x in avgimgpath.rglob('*') if x.is_dir()]
rawavipath = Path.joinpath(searchpath, "RawVideos")
rawavi_subdirs = [x for x in rawavipath.rglob('*') if x.is_dir()]

# creating directory to save augmented data
augpath = Path.joinpath(searchpath, "Augmented_Data")

if os.path.exists(augpath):
    augpath_avgimg = Path.joinpath(augpath, "AveragedImages")
    augpath_rawavi = Path.joinpath(augpath, "RawVideos")
    augpath_avgimg_conf = Path.joinpath(augpath_avgimg, "confocal")
    augpath_avgimg_split = Path.joinpath(augpath_avgimg, "split")
    augpath_rawavi_conf = Path.joinpath(augpath_rawavi, "confocal")
    augpath_rawavi_split = Path.joinpath(augpath_rawavi, "split")

else:
    os.mkdir(augpath)

    augpath_avgimg = Path.joinpath(augpath, "AveragedImages")
    augpath_rawavi = Path.joinpath(augpath, "RawVideos")
    os.mkdir(augpath_avgimg)
    os.mkdir(augpath_rawavi)

    augpath_avgimg_conf = Path.joinpath(augpath_avgimg, "confocal")
    augpath_avgimg_split = Path.joinpath(augpath_avgimg, "split")
    augpath_rawavi_conf = Path.joinpath(augpath_rawavi, "confocal")
    augpath_rawavi_split = Path.joinpath(augpath_rawavi, "split")
    os.mkdir(augpath_avgimg_conf)
    os.mkdir(augpath_avgimg_split)
    os.mkdir(augpath_rawavi_conf)
    os.mkdir(augpath_rawavi_split)

for path in avgimg_subdirs:
    if "confocal" in path.name:
        conf_avgimg_png = [x for x in path.rglob("*.png")]
        conf_avgimg_tif = [x for x in path.rglob("*.tif")]
        conf_avgimg = conf_avgimg_png + conf_avgimg_tif

        for p in conf_avgimg:
            tmpimg = cv2.imread(str(p))
            flippedtmpimg = cv2.flip(tmpimg, 1)

            # TODO: change the coordinate locations for OCVL images in future so they mirror the actual location
            # (ie if temporal make flipped location nasal)... Too tricky to do with current file structure

            img_name = p.name
            try:
                new_name = img_name.replace('.png', '_flipped.png')
            except:
                new_name = img_name.replace('.tif', '_flipped.tif')
            augpath_avgimg_conf_flip = Path.joinpath(augpath_avgimg_conf, new_name)

            # saving flipped images
            cv2.imwrite(str(augpath_avgimg_conf_flip), flippedtmpimg)

            # plt.figure()
            # plt.imshow(flippedtmpimg)
            # plt.show()

    elif "split" in path.name:
        split_avgimg_png = [x for x in path.rglob("*.png")]
        split_avgimg_tif = [x for x in path.rglob("*.tif")]
        split_avgimg = split_avgimg_png + split_avgimg_tif

        for pp in split_avgimg:
            tmpimg_sp = cv2.imread(str(pp))
            flippedtmpimg_sp = cv2.flip(tmpimg_sp, 1)

            # TODO: change the coordinate locations for OCVL images in future so they mirror the actual location
            # (ie if temporal make flipped location nasal)... Too tricky to do with current file structure

            img_name_sp = pp.name
            new_name_sp = img_name_sp.replace('.png', '_flipped.png')
            augpath_avgimg_split_flip = Path.joinpath(augpath_avgimg_split, new_name_sp)

            # saving flipped images
            cv2.imwrite(str(augpath_avgimg_split_flip), flippedtmpimg_sp.)

            # plt.figure()
            # plt.imshow(flippedtmpimg)
            # plt.show()
            #print(' ')


# for path in rawavi_subdirs:
#     if "confocal" in path.name:
#         conf_avi = [x for x in path.rglob("*.avi")]
#
#         for v in conf_avi:
#             test = load_video(str(v))
#
#             print()
#
#
#     if "split" in path.name:
#         print('split')
#
#         if (path.parent.parent == searchpath or path.parent == searchpath):
#             if path.parent not in allFiles:
#                 allFiles[path.parent] = []
#                 allFiles[path.parent].append(path)
#
#                 if "control" in path.parent.name:
#                     # print("DETECTED CONTROL DATA AT: " + str(path.parent))
#                     controlpath = path.parent
#             else:
#                 allFiles[path.parent].append(path)
#
#         totFiles += 1

# TODO: Loop through all images. Flip everything left-right save flipped image "_flippedLR.png"

# TODO: Loop through all avis. Flip all frames left-right save flipped video "_flippedLR.avi"

# TODO: Update observations spreadsheet to match the score from the respective original images + add new filenames
#  (as flipping LR will not impact subjective grading score on cone resolvability)






