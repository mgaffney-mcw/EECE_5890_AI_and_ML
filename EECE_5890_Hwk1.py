# EECE 5890 Hwk 1
# Created by: Mina Gaffney
# Created on: 01/27/2024

# Here is the dataset for Homework #1.
#
# https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
#
# This assignment is about fetching data from csv using Python and exploring the dataset.
#
# 1) Identify how many employee records are stored.
# 2) Identify (count) number of employees based on their education (degrees).

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

# 1) Identify how many employee records are stored
# Finding the length of the dataframe aka the number of rows
numEmployees = dataset.shape[0]
# Find the width of the dataframe aka the number of columns
numEmployeeRecords = dataset.shape[1]
print(' ')
print(str(numEmployeeRecords) + ' employee records are stored in this dataset for ' + str(numEmployees) + ' total employees.')

# 2) Identify (count) number of employees based on their education (degrees).
# Total number of employees was found above by finding the numbers of rows
# Now let's break down the total number based on their education
# First we have to identify which column stores education info
# data is stored in the Education column
# Now let's count the instances of Bachelors, Masters, and PhD:
edCount = dataset['Education'].value_counts()
print(' ')
print('Displaying number of employees based on their education:')
print(edCount)
# Displaying number of employees based on their education in a bar graph
plt.figure
sns.countplot(x='Education', data=dataset)
plt.title('Distribution of employees based on degree earned', fontsize=20)
plt.xlabel('Degree Earned', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.show()
print(' ')
print('Done!')

