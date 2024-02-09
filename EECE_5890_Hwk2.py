# EECE 5890 Hwk 2
# Created by: Mina Gaffney
# Created on: 02/09/2024

# Here is the dataset for Homework #1.
#
# https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
#
# Use the dataset from HW1 and try to solve following questions:
#
# Identify which gender employee is more likely to leave a job.
# Is there any effect on the year of employment while leaving a job?
# Find out cities where employee retention rate is higher than other cities.

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import seaborn as sns

# User selects dataset from file
root = Tk()
fName = filedialog.askopenfilename(title="Select the Kaggle Employee dataset.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# Reading in dataset as dataframe
dataset=pd.read_csv(fName)
print(' ')
print('Dataset loaded.')

# 1) Identify which gender employee is more likely to leave a job.
edCount = dataset['Education'].value_counts()
print(' ')
print('Displaying number of employees based on their education:')
print(edCount)

# 2) Is there any effect on the year of employment while leaving a job?


# 3) Find out cities where employee retention rate is higher than other cities.