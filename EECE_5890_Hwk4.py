# EECE 5890 Hwk 4
# Created by: Mina Gaffney
# Created on: 03/29/2024

# Attached is a dataset of medical papers title and abstract.
#
# You can find details about this dataset on Kaggle.
#
# https://www.kaggle.com/datasets/wolfmedal/medical-paper-title-and-abstract-dataset?resource=download
#
# You need to generate a word cloud for abstract column data using NLP following NLP techniques:
#
# Removal of stop words
# Stemming
# Note: Feel free to reach out to me if you have any questions.

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

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


print('Done!')