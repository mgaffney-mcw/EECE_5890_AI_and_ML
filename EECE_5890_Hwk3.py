# EECE 5890 Hwk 3
# Created by: Mina Gaffney
# Created on: 03/11/2024

# Here is the dataset for Homework #1.
#
# https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
#
# Use the dataset from HW1 and try to solve following questions:
#
# Visualize number of employee joining for each year.
# Is there a correlation between Payment Tier and Experience in Current Domain?

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

# 1) Visualizing the number of employees who have joined each year.
# Visualizing all employees by joining year
plt.figure(1)
countplot = sns.countplot(x = "JoiningYear", data = dataset)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Joining Year")
plt.ylabel("Employee Count")
plt.title("Employee Joining Year", fontweight = "bold")

# Visualizing employee joining year vs gender
plt.figure(2)
countplot = sns.countplot(x = "JoiningYear", hue = "Gender", data = dataset)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Joining Year")
plt.ylabel("Employee Count")
plt.legend(labels = ["Male", "Female"])
plt.title("Employee Joining Year by Gender", fontweight = "bold")

# 2) Is there a correlation between Payment Tier and Experience in Current Domain?
# First visualizing payment tier and experience in current domain
plt.figure(3)
countplot = sns.countplot(x = "ExperienceInCurrentDomain", hue = "PaymentTier", data = dataset)
for count in countplot.containers:
    countplot.bar_label(count,)
plt.xlabel("Experience in Current Domain")
plt.ylabel("Employee Count")
plt.legend(labels = ["1", "2", "3"])
plt.title("Experience in Current Domain by Payment Tier", fontweight = "bold")

# Next showing the correlation matrix
experiencePay_cross_tab = pd.crosstab(dataset['PaymentTier'], dataset['ExperienceInCurrentDomain'])

plt.figure(4)
sns.heatmap(experiencePay_cross_tab, annot=True, cmap='Blues', fmt='d')
plt.title('Correlation Between Payment Tier and Experience in Current Domain')
plt.ylabel('Payment Tier')
plt.xlabel('Experience in Current Domain')

print('Based on the correlation matrix it looks like individuals hired in 2018 are more likely to leave their job.')

plt.show()
print('Done!')