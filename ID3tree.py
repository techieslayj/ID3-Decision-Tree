#--------------------- AJ Iglesias ----------------------------

# - ID3 Decisin Tree algorithm using SKlearn

#The following is a simplified decisin tree machine learning algorithm
#The dataset is my own and small, built on 60 days of 4 features and 1 target (whether to play (1) or dont play on that day (0))
#I include the sklearn decisiontreeclassifier for gini_impurity which is is a measure of how often a randomly
#chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution
#of labels in the subset. Comparing the yprediction and x_test sets show how accurate the ML algorithm is in determining
#how well it predicts based on the original data set. I finish the program by both writing a .txt file whose contents
#can be copied and pastedon webgraphviz.com in order to see the drawn decision tree.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#read in my data set playsoccer.csv delimiter is , since a .csv

data = pd.read_csv('Soccerdataset.csv', delimiter = ',')

#print(data)
#data type print out to ensure data type is accurate (object)
# also print out data set length and shape
#print(data.dtypes)
#print(len(data))
#print(data.shape)

df = pd.DataFrame(data)

#Need to change string values to int values for the DecisionTreeClassifier function from skLearn
#Start with oour outlook column which is the only column to have 3 attributes (overcast, sunny, rain)
df.replace({'Overcast': 2, 'Sunny': 1, 'Rain': 0}, inplace = True, regex = True)
df.replace({'Hot': 2, 'Mild': 1, 'Cool': 0}, inplace = True, regex = True)

#Now make the next columns binary as the next columns all only have 2 attributes
df.replace({'High': 1, 'Normal': 0}, inplace = True, regex = True)
df.replace({'Strong': 1, 'Weak': 0}, inplace = True, regex = True)
df.replace({'Yes': 1, 'No': 0}, inplace = True, regex = True)

#print dataframe if want to make sure we have all integers within our data
cleandata = pd.DataFrame(df)
#print(cleandata)

#Dataset slicing into target variable (y) and feature set (x)
x = cleandata.values[:, 1:5]
y = cleandata.values[:,5]
#print(x)
#print(y)

#split into training and test sets
#I will denote test_size as .3 so that test set will be 30% of the whole data set while the training set is 70%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#so now X_train and y_train are the training sets
#while X_test and y_test are the test sets

#Get Decision Tree Classifier gini importance is below will not be used here becaus this is common to the CART method
# and requires all data within the data set to be binary meaning each column should only have 2 outcomes (i.e T or F)

#clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 100,
                               #max_depth=3, min_samples_leaf=1)
#clf_gini.fit(X_train, y_train)

#Set up Decision Tree Classifier using SkLearn DecisionTreeClassifier using entropy as criterion since doing ID3 tree
clf_entropy = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', min_samples_split = 2,
                            min_samples_leaf = 1)

print('<<<<<<< Trained Decision Tree Classifier >>>>>>>>> \n', clf_entropy)

#fit our data after using DecisionTreeClassifier
clf_entropy.fit(X_train, y_train)

#If one wants to confirm it took a decision path
#path = clf_entropy.decision_path
#print(path)
#Now ready to predict the model using sklearn .predict()
yprediction = clf_entropy.predict(X_test)

#print results
print('\nResults for whether to play soccer or not for 30% of data set, where Yes = 1 and No = 0 :\n')
print('Predicted outcome:\n', yprediction, '\n')
print('Actual Outcome:\n', y_test, '\n')





#Taking accuracy of the model (Tells us how often the trained program will match the outcome or target of our data set
#i.e. whether we should PlaySoccer or not)
print('Accuracy of the Decision Tree ID3 model for this data: ', accuracy_score(y_test, yprediction)*100)

#Confirm the correct data set was used
#print(cleandata)

#convert out classifier tree into a txt file that can be copied to local directory and then its contents can be
#copied and pasted on http://webgraphviz.com and tree can be viewed

with open('playsoccer_Classifier.txt', 'w') as f:
    f = tree.export_graphviz(clf_entropy, out_file = f)

#PROGRAM FINISH
