# EECE 5890 Hwk 5
# Created by: Mina Gaffney
# Created on: 04/21/2024

# For HW5, we will use the same dataset from HW4.
#
# You can find details about this dataset on Kaggle.
#
# https://www.kaggle.com/datasets/wolfmedal/medical-paper-title-and-abstract-dataset?resource=download
#
# You can use NLTK or any other NLP techniques (BERT is encouraged but not required).
# Remove stop words,
# apply lemmatization,
# and generate a token from "Title" and "Abstract" columns.
#
# Once you generate a token from these columns identify
# common tokens and list those tokens, for individual rows (or record).

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import string
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# User selects dataset from file
root = Tk()
fName = filedialog.askopenfilename(title="Select the Kaggle dataset.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# Reading in dataset as dataframe
dataset=pd.read_csv(fName)
print(' ')
print('Dataset loaded.')


# Removing stop words, applying lemmatization, and generating a token from "Title" and "Abstract" columns.
stp_wrds = set(stopwords.words('english'))
pstem = PorterStemmer()

# Pre-allocating things
abs_no_stop = np.empty((len(dataset.abstract), 1))
abs_no_stop_t = np.empty((len(dataset.title), 1))
abs_lem = np.empty((len(dataset.abstract), 1))
abs_lem_t = np.empty((len(dataset.title), 1))
abs_no_stop[:] = np.nan
abs_lem[:] = np.nan
abs_no_stop_t[:] = np.nan
abs_lem_t[:] = np.nan
abs_no_stop = pd.DataFrame(abs_no_stop, columns=['abstract'])
abs_no_stop_t = pd.DataFrame(abs_no_stop_t, columns=['title'])
abs_lem = pd.DataFrame(abs_stem, columns=['abstract'])
abs_lem_t = pd.DataFrame(abs_stem, columns=['title'])
str_abs_stem = ''

count = 0
for i in dataset.abstract:
    i_tkn = word_tokenize(i) #tokenizing in abstract column
    i_no_stop = [w for w in i_tkn if not w.lower() in stp_wrds]
    i_lem = [wnl.lemmatize(n, pos='v') for n in i_no_stop]
    i_lem_str = ' '.join(i_lem)
    i_lem_str = i_lem_str.translate(i_lem_str.maketrans('', '', string.punctuation))
    abs_no_stop.loc[count] = [[i_no_stop]]
    abs_lem.loc[count] = [[i_lem]]

    # Storing no stop stemmed words in one giant list:
    str_abs_lem = str_abs_lem + i_lem_str
    count = count + 1

count2 = 0
for t in dataset.title:
    t_tkn = word_tokenize(t) #tokenizing in abstract column
    t_no_stop = [ww for ww in t_tkn if not ww.lower() in stp_wrds]
    t_lem = [wnl.lemmatize(nn, pos='v') for nn in t_no_stop]
    t_lem_str = ' '.join(t_lem)
    t_lem_str = t_lem_str.translate(t_lem_str.maketrans('', '', string.punctuation))
    abs_no_stop_t.loc[count] = [[t_no_stop]]
    abs_lem_t.loc[count] = [[t_lem]]

    # Storing no stop stemmed words in one giant list:
    str_abs_lem_t = str_abs_lem_t + t_lem_str
    count2 = count2 + 1

# Once you generate a token from these columns identify common tokens and list those tokens, for individual rows (or record).
