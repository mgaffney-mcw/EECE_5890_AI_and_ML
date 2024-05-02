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

# TODO: Load in folders containing the confocal and split images and videos
#  (4 separate folders based on current data organization structure)

# TODO: Loop through all images. Flip everything left-right save flipped image "_flippedLR.png"

# TODO: Loop through all avis. Flip all frames left-right save flipped video "_flippedLR.avi"




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

