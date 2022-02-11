#GEOG 313 Python classification script
#Import all the libraries
import numpy as np
import matplotlib
import csv
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

#import CloudCompare points ASCII File as CSV
data = pd.read_csv("classes_final.csv")
df = pd.DataFrame(data)

#define classes
landClass = df['class'].tolist()
rows = df.values.tolist()

x = df['x'].tolist()
y = df['y'].tolist()
z = df['z'].tolist()
r = df['r'].tolist()
g = df['g'].tolist()
b = df['b'].tolist()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Part 2: Implement Random Forest
cloudData = pd.read_csv("ptcloud_final.csv")
df2 = pd.DataFrame(cloudData)
feature_names = ['x', ' y', ' z', ' r', ' g', ' b']

#H = training data columns, G = desired prediction column(s)
H = data[['x', 'y', 'z', 'r', 'g', 'b']]
G = data['class']

#Cloudclass = entire cloud dataset
cloudclass = cloudData[['x', ' y', ' z', ' r', ' g', ' b']]

#Train the classifier with training data
#Test size = 0.3: 30% training points, 70% testing points
H_train, H_test, G_train, G_test = train_test_split(H, G, test_size = 0.3)

#Create Gaussian classifier
gauss_clf = RandomForestClassifier(n_estimators = 100)
gauss_clf.fit(H_train, G_train)

#Predict classes of test data and determine accuracy of classifier
G_pred = gauss_clf.predict(H_test)
print("Accuracy:",metrics.accuracy_score(G_test, G_pred))

#Apply Classifier to entire point cloud
cloud_pred = gauss_clf.predict(cloudclass)

#Append predicted class as a column to point cloud data
df2['Predicted_Class'] = cloud_pred

#Write new .csv to upload to CloudCompare as point cloud in XYZRGB format
df2.to_csv('randomforest_final.csv')
