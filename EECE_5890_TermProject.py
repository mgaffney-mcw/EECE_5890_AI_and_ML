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

from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

from pathlib import Path
import os
import warnings
from enum import Enum
from os.path import exists, splitext
import cv2
import numpy as np

## Loading in labels + training + augmented data
# User selects dataset from file
root = Tk()
fName = filedialog.askopenfilename(title="Select Brennan_AIQManuscriptDataSpreadsheet.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# Reading in dataset as dataframe
dataset=pd.read_excel(fName, sheet_name=['AOIP','Testdata'])
AOIP_df=dataset['AOIP']
#OCVL_df=dataset['OCVL']
AOIP_test_df=dataset['Testdata']


print(' ')
print('Dataset loaded.')

# Note: this will be written to expect a very specific folder structure as follows:
#   Lvl 1: Training_Data
#       Lvl 2: Averaged Images
#           Lvl 3: confocal - averaged confocal tifs or pngs
#           Lvl 3: split - averaged split tifs or pngs
#       Lvl 2: RawVideos
#           Lvl 3: confocal - raw confocal avis
#           Lvl 3: split - raw split avis

# Loading in folder directory which contains all training + augmented data (contains subfolders with images and avis)
pName2 = filedialog.askdirectory(title="Select the folder containing all raw videos and averaged images of interest.", parent=root)

if not pName2:
    quit()

pName3 = filedialog.askdirectory(title="Select the folder containing all the test averaged images of interest.", parent=root)

if not pName3:
    quit()

# defining path names according to above folder structure
searchpath = Path(pName2)
avgimgpath = Path.joinpath(searchpath, "AveragedImages")
avgimg_subdirs = [x for x in avgimgpath.rglob('*') if x.is_dir()]
rawavipath = Path.joinpath(searchpath, "RawVideos")
rawavi_subdirs = [x for x in rawavipath.rglob('*') if x.is_dir()]
augpath = Path.joinpath(searchpath, "Augmented_Data")
augpath_avgimg = Path.joinpath(augpath, "AveragedImages")
augpath_rawavi = Path.joinpath(augpath, "RawVideos")
augpath_avgimg_conf = Path.joinpath(augpath_avgimg, "confocal")
augpath_avgimg_split = Path.joinpath(augpath_avgimg, "split")
augpath_rawavi_conf = Path.joinpath(augpath_rawavi, "confocal")
augpath_rawavi_split = Path.joinpath(augpath_rawavi, "split")

searchpath_test = Path(pName3)
avgimgpath_test = Path.joinpath(searchpath_test, "AveragedImages")
avgimg_subdirs_test = [x for x in avgimgpath_test.rglob('*') if x.is_dir()]
rawavipath_test = Path.joinpath(searchpath_test, "RawVideos")
rawavi_subdirs_test= [x for x in rawavipath_test.rglob('*') if x.is_dir()]


## Cleaning labeled dataset
# Checking for duplicate filenames in the Confocal and Split columns
duplicate_AOIP_Conf = AOIP_df['Confocal Image name'].duplicated()
duplicate_AOIP_Split = AOIP_df['Split Image name'].duplicated()
#duplicate_OCVL_Conf = OCVL_df['Confocal Image name'].duplicated()
#duplicate_OCVL_Split = OCVL_df['Split Image name'].duplicated()

# Deleting any duplicate values
# cleaned_AOIP_df = AOIP_df.drop_duplicates(subset=['Confocal Image name'], keep='first')
# cleaned_AOIP_df = AOIP_df.drop_duplicates(subset=['Split Image name'], keep='first')

cleaned_AOIP_df = AOIP_df.copy()
cleaned_test_df = AOIP_test_df.copy()

#cleaned_OCVL_df = OCVL_df.drop_duplicates(subset=['Confocal Image name'], keep='first')
#cleaned_OCVL_df = OCVL_df.drop_duplicates(subset=['Split Image name'], keep='first')

# Rounding SNR value and Average to two decimal places
cleaned_AOIP_df = cleaned_AOIP_df.round({'Confocal SNR value': 2, 'Confocal Average Grade': 2, 'Split SNR value': 2, 'Split Average Grade': 2})
cleaned_test_df = cleaned_test_df.round({'Confocal SNR value': 2, 'Confocal Average Grade': 2, 'Split SNR value': 2, 'Split Average Grade': 2})
#cleaned_OCVL_df = cleaned_OCVL_df.round({'Confocal SNR value': 2, 'Confocal Average Grade': 2, 'Split SNR value': 2, 'Split Average Grade': 2})

# Separating out Confocal and Split modalities
AOIP_confocal = cleaned_AOIP_df[['Confocal Image name','Confocal SNR value','Confocal Grader 1','Confocal Grader 2','Confocal Grader 3','Confocal Average Grade']]
AOIP_Split = cleaned_AOIP_df[['Split Image name','Split SNR value','Split Grader 1','Split Grader 2','Split Grader 3','Split Average Grade']]

AOIP_confocal_test = cleaned_test_df[['Confocal Image name','Confocal SNR value','Confocal Grader 1','Confocal Grader 2','Confocal Grader 3','Confocal Average Grade']]
AOIP_Split_test = cleaned_test_df[['Split Image name','Split SNR value','Split Grader 1','Split Grader 2','Split Grader 3','Split Average Grade']]
#OCVL_confocal = cleaned_OCVL_df[['Confocal Image name','Confocal SNR value','Confocal Grader 1','Confocal Grader 2','Confocal Grader 3','Confocal Average Grade']]
#OCVL_Split = cleaned_OCVL_df[['Split Image name','Split SNR value','Split Grader 1','Split Grader 2','Split Grader 3','Split Average Grade']]

# Checking that there are no missing values
AOIP_confocal_notnull = AOIP_confocal[pd.notnull(AOIP_confocal)]
AOIP_Split_notnull = AOIP_Split[pd.notnull(AOIP_Split)]
AOIP_test_notnull = AOIP_confocal_test[pd.notnull(AOIP_confocal_test)]

#OCVL_confocal_notnull = OCVL_confocal[pd.notnull(OCVL_confocal)]
#OCVL_Split_notnull = OCVL_Split[pd.notnull(OCVL_Split)]

# Summarizing data
# Showing boxplots of all SNR values
# box_data_SNR = [AOIP_confocal_notnull['Confocal SNR value'], AOIP_Split_notnull['Split SNR value'], OCVL_confocal_notnull['Confocal SNR value'], OCVL_Split_notnull['Split SNR value']]
# box_ticks = ['AOIP Confocal', 'AOIP Split', 'OCVL Confocal', 'OCVL Split']
# fig1, ax1 = plt.subplots()
# ax1.set_title('SNR Values')
# ax1.set_xticklabels(box_ticks)
# plt.ylabel("SNR value")
# ax1.boxplot(box_data_SNR)


# Showing boxplot of average Grade
# box_data_AvgGrade = [AOIP_confocal_notnull['Confocal Average Grade'], AOIP_Split_notnull['Split Average Grade'], OCVL_confocal_notnull['Confocal Average Grade'], OCVL_Split_notnull['Split Average Grade']]
# fig2, ax2 = plt.subplots()
# ax2.set_title('Average Grade')
# ax2.set_xticklabels(box_ticks)
# plt.ylabel("Average Grade")
# ax2.boxplot(box_data_AvgGrade)


# Reorganizing data so I can visualize with countplot etc
AOIP_confocal_notnull['Modality'] = 0 #Confocal modality = 0
AOIP_Split_notnull['Modality'] = 1 #Split modality = 1
AOIP_confocal_notnull['Location'] = 0 #AOIP imaging location = 0
AOIP_Split_notnull['Location'] = 0 #AOIP imaging location = 0

# OCVL_confocal_notnull['Modality'] = 0 #Confocal modality = 0
# OCVL_Split_notnull['Modality'] = 1 #Split modality = 1
# OCVL_confocal_notnull['Location'] = 1 #OCVL imaging location = 1
# OCVL_Split_notnull['Location'] = 1 #OCVL imaging location = 1

All_comb_df = AOIP_confocal_notnull[['Confocal Image name', 'Confocal SNR value', 'Confocal Grader 1','Confocal Average Grade', 'Modality', 'Location']]
All_comb_df['Grader'] = 1 #Grader 1 = 1
All_comb_df = All_comb_df.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 1':'Grade','Confocal Average Grade':'Average Grade'})

AOIP_conf_G2_tmp = AOIP_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 3'], axis=1)
AOIP_conf_G2_tmp['Grader'] = 2 #Grader 2 = 2
AOIP_conf_G2_tmp = AOIP_conf_G2_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 2':'Grade','Confocal Average Grade':'Average Grade'})

AOIP_conf_G3_tmp = AOIP_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 2'], axis=1)
AOIP_conf_G3_tmp['Grader'] = 3 #Grader 3 = 3
AOIP_conf_G3_tmp = AOIP_conf_G3_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 3':'Grade','Confocal Average Grade':'Average Grade'})

AOIP_split_G1_tmp = AOIP_Split_notnull.drop(['Split Grader 2','Split Grader 3'], axis=1)
AOIP_split_G1_tmp['Grader'] = 1 #Grader 1 = 1
AOIP_split_G1_tmp = AOIP_split_G1_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 1':'Grade','Split Average Grade':'Average Grade'})

AOIP_split_G2_tmp = AOIP_Split_notnull.drop(['Split Grader 1','Split Grader 3'], axis=1)
AOIP_split_G2_tmp['Grader'] = 2 #Grader 2 = 2
AOIP_split_G2_tmp = AOIP_split_G2_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 2':'Grade','Split Average Grade':'Average Grade'})

AOIP_split_G3_tmp = AOIP_Split_notnull.drop(['Split Grader 1','Split Grader 2'], axis=1)
AOIP_split_G3_tmp['Grader'] = 3 #Grader 3 = 3
AOIP_split_G3_tmp = AOIP_split_G3_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 3':'Grade','Split Average Grade':'Average Grade'})

# OCVL_conf_G1_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 2','Confocal Grader 3'], axis=1)
# OCVL_conf_G1_tmp['Grader'] = 1 #Grader 1 = 1
# OCVL_conf_G1_tmp = OCVL_conf_G1_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 1':'Grade','Confocal Average Grade':'Average Grade'})
#
# OCVL_conf_G2_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 3'], axis=1)
# OCVL_conf_G2_tmp['Grader'] = 2 #Grader 2 = 2
# OCVL_conf_G2_tmp = OCVL_conf_G2_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 2':'Grade','Confocal Average Grade':'Average Grade'})
#
# OCVL_conf_G3_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 2'], axis=1)
# OCVL_conf_G3_tmp['Grader'] = 3 #Grader 3 = 3
# OCVL_conf_G3_tmp = OCVL_conf_G3_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 3':'Grade','Confocal Average Grade':'Average Grade'})
#
# OCVL_split_G1_tmp = OCVL_Split_notnull.drop(['Split Grader 2','Split Grader 3'], axis=1)
# OCVL_split_G1_tmp['Grader'] = 1 #Grader 1 = 1
# OCVL_split_G1_tmp = OCVL_split_G1_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 1':'Grade','Split Average Grade':'Average Grade'})
#
# OCVL_split_G2_tmp = OCVL_Split_notnull.drop(['Split Grader 1','Split Grader 3'], axis=1)
# OCVL_split_G2_tmp['Grader'] = 2 #Grader 2 = 2
# OCVL_split_G2_tmp = OCVL_split_G2_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 2':'Grade','Split Average Grade':'Average Grade'})
#
# OCVL_split_G3_tmp = OCVL_Split_notnull.drop(['Split Grader 1','Split Grader 2'], axis=1)
# OCVL_split_G3_tmp['Grader'] = 3 #Grader 3 = 3
# OCVL_split_G3_tmp = OCVL_split_G3_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 3':'Grade','Split Average Grade':'Average Grade'})

All_comb_df = pd.concat([All_comb_df, AOIP_conf_G2_tmp, AOIP_conf_G3_tmp, AOIP_split_G1_tmp, AOIP_split_G2_tmp, AOIP_split_G3_tmp], ignore_index=True)
 #OCVL_conf_G1_tmp, OCVL_conf_G2_tmp, OCVL_conf_G3_tmp, OCVL_split_G1_tmp, OCVL_split_G2_tmp, OCVL_split_G3_tmp], ignore_index=True)

# for now, we are just going to round the average grade and use that as our label
# TODO restructure image data frame to be able to use the All_comb_df for labels

AOIP_conf_avg_label = AOIP_confocal_notnull['Confocal Average Grade']
AOIP_conf_test_label = AOIP_test_notnull['Confocal Average Grade']
#OCVL_conf_avg_label = OCVL_confocal_notnull['Confocal Average Grade']

AOIP_split_avg_label = AOIP_Split_notnull['Split Average Grade']
#OCVL_split_avg_label = OCVL_Split_notnull['Split Average Grade']

# all_conf_labels = pd.concat([AOIP_conf_avg_label, OCVL_conf_avg_label], ignore_index=True)
# all_conf_labels = round(all_conf_labels).astype(int)
# all_split_labels = pd.concat([AOIP_split_avg_label, OCVL_split_avg_label], ignore_index=True)
# all_split_labels = round(all_split_labels).astype(int)

all_conf_labels = AOIP_conf_avg_label.copy()
all_conf_labels = round(all_conf_labels).astype(int)
all_split_labels = AOIP_split_avg_label
all_split_labels = round(all_split_labels).astype(int)
all_test_labels = AOIP_conf_test_label.copy()
all_test_labels = round(all_test_labels).astype(int)

all_conf_labels = round(all_conf_labels).astype(int)
all_split_labels = round(all_split_labels).astype(int)

## Image processing to ensure all images are square, of the same dimension and range from 0-1

for path in avgimg_subdirs:
    if "confocal" in path.name:
        conf_avgimg_png = [x for x in path.rglob("*.png")]
        conf_avgimg_tif = [x for x in path.rglob("*.tif")]
        conf_avgimg = conf_avgimg_png + conf_avgimg_tif

        #all_conf_images = np.empty([720, 720, len(conf_avgimg)])
        all_conf_images = np.empty([int(len(conf_avgimg)), 720, 720])


        counter = 0
        for p in conf_avgimg:
            tmpimg = cv2.imread(str(p))

            height, width, channels = tmpimg.shape
            print(height, width, channels)

            if height == width:
                if height == 720:
                    resize_img = tmpimg[0:720, 0:720, 1]
                else:
                    crop_img = tmpimg[0:int(height), 0:int(width),1]
                    resize_img = cv2.resize(crop_img, (720, 720),
                                            interpolation=cv2.INTER_LINEAR)
            elif height < width:
                crop = (width-height)/2
                crop_img = tmpimg[0:int(height), int(crop):int(width-crop),1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                          interpolation=cv2.INTER_LINEAR)
            elif width < height:
                crop = (height - width) / 2
                crop_img = tmpimg[int(crop):int(height - crop), 0:int(width), 1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                        interpolation=cv2.INTER_LINEAR)

            all_conf_images[counter, 0:720, 0:720] = resize_img
            counter = counter+1

    elif "split" in path.name:
        split_avgimg_png = [x for x in path.rglob("*.png")]
        split_avgimg_tif = [x for x in path.rglob("*.tif")]
        split_avgimg = split_avgimg_png + split_avgimg_tif

        all_split_images = np.empty([len(split_avgimg), 720, 720])

        counter2 = 0
        for pp in split_avgimg:
            tmpimg = cv2.imread(str(pp))

            height, width, channels = tmpimg.shape
            print(height, width, channels)

            if height == width:
                if height == 720:
                    resize_img = tmpimg[0:720, 0:720, 1]
                else:
                    crop_img = tmpimg[0:int(height), 0:int(width), 1]
                    resize_img = cv2.resize(crop_img, (720, 720),
                                            interpolation=cv2.INTER_LINEAR)
            elif height < width:
                crop = (width - height) / 2
                crop_img = tmpimg[0:int(height), int(crop):int(width - crop), 1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                        interpolation=cv2.INTER_LINEAR)
            elif width < height:
                crop = (height - width) / 2
                crop_img = tmpimg[int(crop):int(height - crop), 0:int(width), 1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                        interpolation=cv2.INTER_LINEAR)

            all_split_images[counter2, 0:720, 0:720] = resize_img
            counter2 = counter2 + 1


for path in avgimg_subdirs_test:
    if "confocal" in path.name:
        conf_avgimg_test_png = [x for x in path.rglob("*.png")]
        conf_avgimg_test_tif = [x for x in path.rglob("*.tif")]
        conf_avgimg_test = conf_avgimg_test_png + conf_avgimg_test_tif

        #all_conf_images = np.empty([720, 720, len(conf_avgimg)])
        all_conf_images_test = np.empty([int(len(conf_avgimg_test)), 720, 720])


        counter = 0
        for p in conf_avgimg_test:
            tmpimg = cv2.imread(str(p))

            height, width, channels = tmpimg.shape
            print(height, width, channels)

            if height == width:
                if height == 720:
                    resize_img = tmpimg[0:720, 0:720, 1]
                else:
                    crop_img = tmpimg[0:int(height), 0:int(width),1]
                    resize_img = cv2.resize(crop_img, (720, 720),
                                            interpolation=cv2.INTER_LINEAR)
            elif height < width:
                crop = (width-height)/2
                crop_img = tmpimg[0:int(height), int(crop):int(width-crop),1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                          interpolation=cv2.INTER_LINEAR)
            elif width < height:
                crop = (height - width) / 2
                crop_img = tmpimg[int(crop):int(height - crop), 0:int(width), 1]
                resize_img = cv2.resize(crop_img, (720, 720),
                                        interpolation=cv2.INTER_LINEAR)

            all_conf_images_test[counter, 0:720, 0:720] = resize_img
            counter = counter+1


## splitting dataset into training vs

# split manually
all_conf_images = all_conf_images / 255
all_split_images = all_split_images / 255

## Building model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(720, 720)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6)])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(all_conf_images, all_conf_labels, epochs=6)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

test_loss, test_acc = model.evaluate(all_conf_images_test,  all_test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = probability_model.predict(all_conf_images_test)

predictions[0]

np.argmax(predictions[0])

all_test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(5))
  plt.yticks([])
  thisplot = plt.bar(range(5), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# Showing bargraph of individual grades by grader
plt.figure(3)
countplot = sns.countplot(x = "Grade", hue = "Grader", data = All_comb_df)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Grade")
plt.ylabel("Number of images")
plt.legend(labels = ["Grader 1", "Grader 2", "Grader 3"])
plt.title("Grades by grader", fontweight = "bold")

# Showing bargraph of individual grades by location
plt.figure(4)
countplot = sns.countplot(x = "Grade", hue = "Location", data = All_comb_df, palette="flare")
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Grade")
plt.ylabel("Number of images")
plt.legend(labels = ["AOIP", "OCVL"])
plt.title("Grades by Location", fontweight = "bold")

# Showing bargraph of individual grades by Modality
plt.figure(5)
countplot = sns.countplot(x = "Grade", hue = "Modality", data = All_comb_df, palette="pastel")
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Grade")
plt.ylabel("Number of images")
plt.legend(labels = ["Confocal", "Split"])
plt.title("Grades by Modality", fontweight = "bold")

# Showing bargraph of SNR by Modality
plt.figure(6)
countplot = sns.histplot(x = "SNR Val", hue = "Modality", data = All_comb_df, palette="pastel")
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("SNR Value")
plt.ylabel("Number of images")
plt.legend(labels = ["Confocal", "Split"])
plt.title("SNR Value by Modality", fontweight = "bold")

# Showing bargraph of individual grades by location
plt.figure(7)
countplot = sns.histplot(x = "SNR Val", hue = "Location", data = All_comb_df, palette="flare")
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("SNR Value")
plt.ylabel("Number of images")
plt.legend(labels = ["AOIP", "OCVL"])
plt.title("SNR Value by Location", fontweight = "bold")

plt.figure(9)
plt.scatter(All_comb_df['SNR Val'], All_comb_df['Average Grade'])
plt.ylabel("Average Grade")
plt.xlabel("SNR Value")
plt.title("Average Grade vs SNR Value", fontweight = "bold")

reg = LinearRegression().fit(All_comb_df['SNR Val'], All_comb_df['Average Grade'])
reg.score(All_comb_df['SNR Val'], All_comb_df['Average Grade'])



plt.show()