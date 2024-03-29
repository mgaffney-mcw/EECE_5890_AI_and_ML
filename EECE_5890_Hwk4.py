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
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import string

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

# Removing stop words and stemming words from the abstract column
stp_wrds = set(stopwords.words('english'))
pstem = PorterStemmer()

# Pre-allocating things
abs_no_stop = np.empty((len(dataset.abstract), 1))
abs_stem = np.empty((len(dataset.abstract), 1))
abs_no_stop[:] = np.nan
abs_stem[:] = np.nan
abs_no_stop = pd.DataFrame(abs_no_stop, columns=['abstract'])
abs_stem = pd.DataFrame(abs_stem, columns=['abstract'])
str_abs_stem = ''

count = 0
for i in dataset.abstract:
    i_tkn = word_tokenize(i) #tokenizing in abstract column
    i_no_stop = [w for w in i_tkn if not w.lower() in stp_wrds]
    i_stem = [pstem.stem(n) for n in i_no_stop]
    i_stem_str = ' '.join(i_stem)
    i_stem_str = i_stem_str.translate(i_stem_str.maketrans('', '', string.punctuation))
    abs_no_stop.loc[count] = [[i_no_stop]]
    abs_stem.loc[count] = [[i_stem]]
    # Storing no stop stemmed words in one giant list:
    str_abs_stem = str_abs_stem + i_stem_str
    count = count + 1

# Generate a word cloud image
wordcloud = WordCloud(font_path = 'C:\Windows\Fonts\Arial.ttf').generate(str_abs_stem)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.show()
print('Done!')