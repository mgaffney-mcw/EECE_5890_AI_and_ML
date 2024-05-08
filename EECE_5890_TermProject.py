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

## Loading in labels + training + augmented data
# User selects dataset from file
root = Tk()
fName = filedialog.askopenfilename(title="Select Brennan_AIQManuscriptDataSpreadsheet.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# Reading in dataset as dataframe
dataset=pd.read_excel(fName, sheet_name=['AOIP','OCVL'])
AOIP_df=dataset['AOIP']
OCVL_df=dataset['OCVL']

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

if not pName:
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


# for path in avgimg_subdirs:
#     if "confocal" in path.name:
#         conf_avgimg_png = [x for x in path.rglob("*.png")]
#         conf_avgimg_tif = [x for x in path.rglob("*.tif")]
#         conf_avgimg = conf_avgimg_png + conf_avgimg_tif
#
#         for p in conf_avgimg:
#             tmpimg = cv2.imread(str(p))
#
#
#
#             # saving flipped images
#             cv2.imwrite(str(augpath_avgimg_conf_flip), flippedtmpimg)
#
#             # plt.figure()
#             # plt.imshow(flippedtmpimg)
#             # plt.show()
#
#     elif "split" in path.name:
#         split_avgimg_png = [x for x in path.rglob("*.png")]
#         split_avgimg_tif = [x for x in path.rglob("*.tif")]
#         split_avgimg = split_avgimg_png + split_avgimg_tif
#
#         for pp in split_avgimg:
#             tmpimg_sp = cv2.imread(str(pp))
#             flippedtmpimg_sp = cv2.flip(tmpimg_sp, 1)
#
#             # TODO: change the coordinate locations for OCVL images in future so they mirror the actual location
#             # (ie if temporal make flipped location nasal)... Too tricky to do with current file structure
#
#             img_name_sp = pp.name
#             new_name_sp = img_name_sp.replace('.png', '_flipped.png')
#             augpath_avgimg_split_flip = Path.joinpath(augpath_avgimg_split, new_name_sp)
#
#             # saving flipped images
#             cv2.imwrite(str(augpath_avgimg_split_flip), flippedtmpimg_sp)
#
#             # plt.figure()
#             # plt.imshow(flippedtmpimg)
#             # plt.show()
#             #print(' ')


## Cleaning labeled dataset
# Checking for duplicate filenames in the Confocal and Split columns
duplicate_AOIP_Conf = AOIP_df['Confocal Image name'].duplicated()
duplicate_AOIP_Split = AOIP_df['Split Image name'].duplicated()
duplicate_OCVL_Conf = OCVL_df['Confocal Image name'].duplicated()
duplicate_OCVL_Split = OCVL_df['Split Image name'].duplicated()

# Deleting any duplicate values
cleaned_AOIP_df = AOIP_df.drop_duplicates(subset=['Confocal Image name'], keep='first')
cleaned_AOIP_df = AOIP_df.drop_duplicates(subset=['Split Image name'], keep='first')

cleaned_OCVL_df = OCVL_df.drop_duplicates(subset=['Confocal Image name'], keep='first')
cleaned_OCVL_df = OCVL_df.drop_duplicates(subset=['Split Image name'], keep='first')

# Rounding SNR value and Average to two decimal places
cleaned_AOIP_df = cleaned_AOIP_df.round({'Confocal SNR value': 2, 'Confocal Average Grade': 2, 'Split SNR value': 2, 'Split Average Grade': 2})
cleaned_OCVL_df = cleaned_OCVL_df.round({'Confocal SNR value': 2, 'Confocal Average Grade': 2, 'Split SNR value': 2, 'Split Average Grade': 2})

# Separating out Confocal and Split modalities
AOIP_confocal = cleaned_AOIP_df[['Confocal Image name','Confocal SNR value','Confocal Grader 1','Confocal Grader 2','Confocal Grader 3','Confocal Average Grade']]
AOIP_Split = cleaned_AOIP_df[['Split Image name','Split SNR value','Split Grader 1','Split Grader 2','Split Grader 3','Split Average Grade']]
OCVL_confocal = cleaned_OCVL_df[['Confocal Image name','Confocal SNR value','Confocal Grader 1','Confocal Grader 2','Confocal Grader 3','Confocal Average Grade']]
OCVL_Split = cleaned_OCVL_df[['Split Image name','Split SNR value','Split Grader 1','Split Grader 2','Split Grader 3','Split Average Grade']]

# Checking that there are no missing values
AOIP_confocal_notnull = AOIP_confocal[pd.notnull(AOIP_confocal)]
AOIP_Split_notnull = AOIP_Split[pd.notnull(AOIP_Split)]
OCVL_confocal_notnull = OCVL_confocal[pd.notnull(OCVL_confocal)]
OCVL_Split_notnull = OCVL_Split[pd.notnull(OCVL_Split)]

# Summarizing data
# Showing boxplots of all SNR values
box_data_SNR = [AOIP_confocal_notnull['Confocal SNR value'], AOIP_Split_notnull['Split SNR value'], OCVL_confocal_notnull['Confocal SNR value'], OCVL_Split_notnull['Split SNR value']]
box_ticks = ['AOIP Confocal', 'AOIP Split', 'OCVL Confocal', 'OCVL Split']
fig1, ax1 = plt.subplots()
ax1.set_title('SNR Values')
ax1.set_xticklabels(box_ticks)
plt.ylabel("SNR value")
ax1.boxplot(box_data_SNR)


# Showing boxplot of average Grade
box_data_AvgGrade = [AOIP_confocal_notnull['Confocal Average Grade'], AOIP_Split_notnull['Split Average Grade'], OCVL_confocal_notnull['Confocal Average Grade'], OCVL_Split_notnull['Split Average Grade']]
fig2, ax2 = plt.subplots()
ax2.set_title('Average Grade')
ax2.set_xticklabels(box_ticks)
plt.ylabel("Average Grade")
ax2.boxplot(box_data_AvgGrade)


# Reorganizing data so I can visualize with countplot etc
AOIP_confocal_notnull['Modality'] = 0 #Confocal modality = 0
AOIP_Split_notnull['Modality'] = 1 #Split modality = 1
AOIP_confocal_notnull['Location'] = 0 #AOIP imaging location = 0
AOIP_Split_notnull['Location'] = 0 #AOIP imaging location = 0

OCVL_confocal_notnull['Modality'] = 0 #Confocal modality = 0
OCVL_Split_notnull['Modality'] = 1 #Split modality = 1
OCVL_confocal_notnull['Location'] = 1 #OCVL imaging location = 1
OCVL_Split_notnull['Location'] = 1 #OCVL imaging location = 1

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

OCVL_conf_G1_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 2','Confocal Grader 3'], axis=1)
OCVL_conf_G1_tmp['Grader'] = 1 #Grader 1 = 1
OCVL_conf_G1_tmp = OCVL_conf_G1_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 1':'Grade','Confocal Average Grade':'Average Grade'})

OCVL_conf_G2_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 3'], axis=1)
OCVL_conf_G2_tmp['Grader'] = 2 #Grader 2 = 2
OCVL_conf_G2_tmp = OCVL_conf_G2_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 2':'Grade','Confocal Average Grade':'Average Grade'})

OCVL_conf_G3_tmp = OCVL_confocal_notnull.drop(['Confocal Grader 1','Confocal Grader 2'], axis=1)
OCVL_conf_G3_tmp['Grader'] = 3 #Grader 3 = 3
OCVL_conf_G3_tmp = OCVL_conf_G3_tmp.rename(columns={'Confocal Image name':'Image Name', 'Confocal SNR value':'SNR Val','Confocal Grader 3':'Grade','Confocal Average Grade':'Average Grade'})

OCVL_split_G1_tmp = OCVL_Split_notnull.drop(['Split Grader 2','Split Grader 3'], axis=1)
OCVL_split_G1_tmp['Grader'] = 1 #Grader 1 = 1
OCVL_split_G1_tmp = OCVL_split_G1_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 1':'Grade','Split Average Grade':'Average Grade'})

OCVL_split_G2_tmp = OCVL_Split_notnull.drop(['Split Grader 1','Split Grader 3'], axis=1)
OCVL_split_G2_tmp['Grader'] = 2 #Grader 2 = 2
OCVL_split_G2_tmp = OCVL_split_G2_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 2':'Grade','Split Average Grade':'Average Grade'})

OCVL_split_G3_tmp = OCVL_Split_notnull.drop(['Split Grader 1','Split Grader 2'], axis=1)
OCVL_split_G3_tmp['Grader'] = 3 #Grader 3 = 3
OCVL_split_G3_tmp = OCVL_split_G3_tmp.rename(columns={'Split Image name':'Image Name', 'Split SNR value':'SNR Val','Split Grader 3':'Grade','Split Average Grade':'Average Grade'})

All_comb_df = pd.concat([All_comb_df, AOIP_conf_G2_tmp, AOIP_conf_G3_tmp, AOIP_split_G1_tmp, AOIP_split_G2_tmp, AOIP_split_G3_tmp, OCVL_conf_G1_tmp, OCVL_conf_G2_tmp, OCVL_conf_G3_tmp, OCVL_split_G1_tmp, OCVL_split_G2_tmp, OCVL_split_G3_tmp])

# multi-class classification with Keras
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("iris.data", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


## Building model ##
# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(720, 720, 3)) # first testing with just ocvl confocal images

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)


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