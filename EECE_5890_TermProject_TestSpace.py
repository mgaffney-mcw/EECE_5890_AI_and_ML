# EECE 5890 Term Project
# Created by: Mina Gaffney
# Created on: 03/05/2024

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