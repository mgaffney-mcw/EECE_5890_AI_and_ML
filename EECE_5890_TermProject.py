# EECE 5890 Term Project
# Created by: Mina Gaffney
# Created on: 03/05/2024

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import seaborn as sns

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

# Cleaning dataset
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
ax1.boxplot(box_data_SNR)


# Showing boxplot of average Grade
box_data_AvgGrade = [AOIP_confocal_notnull['Confocal Average Grade'], AOIP_Split_notnull['Split Average Grade'], OCVL_confocal_notnull['Confocal Average Grade'], OCVL_Split_notnull['Split Average Grade']]
fig2, ax2 = plt.subplots()
ax2.set_title('Average Grade')
ax2.set_xticklabels(box_ticks)
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


# Showing bargraph of individual grades
countplot = sns.countplot(x = "Grade", hue = "Grader", data = All_comb_df)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Grade")
plt.ylabel("Number of images")
plt.legend(labels = ["Grader 1", "Grader 2", "Grader 3"])
plt.title("Number of images with each grade", fontweight = "bold")




plt.show()