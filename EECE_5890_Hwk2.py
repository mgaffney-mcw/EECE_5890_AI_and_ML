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

# First let's take a look at the distribution of employees who stay or leave vs. gender
countplot = sns.countplot(x = "Gender", hue = "LeaveOrNot", data = dataset)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Gender")
plt.ylabel("Employee Count")
plt.legend(labels = ["Stay", "Leave"])
plt.title("Employee Turnover by Gender", fontweight = "bold")

# We can also look at the correlation between gender and leave or not
cross_tab = pd.crosstab(dataset['Gender'], dataset['LeaveOrNot'])

# Since there are more men in this dataset than women I think it makes more sense to normalize the counts
# based on the total number of men and women and express as the percentage or proportion instead
genderCount = dataset['Gender'].value_counts()
TotalMale = genderCount.Male
TotalFemale = genderCount.Female

Norm_cross_tab = cross_tab.div([TotalFemale, TotalMale], axis = 'index')

plt.figure(2)
sns.heatmap(Norm_cross_tab, annot=True, cmap='Blues')
plt.title('Normalized Correlation Between Gender and Employee Turnonver')
plt.xlabel('LeaveOrNot')
plt.ylabel('Gender')
plt.show()

print('Based on the correlation matrix it looks like women are more likely to leave their job.')


# 2) Is there any effect on the year of employment while leaving a job?


# 3) Find out cities where employee retention rate is higher than other cities.