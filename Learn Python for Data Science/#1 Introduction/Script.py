from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
    [154, 54, 37], [166, 65, 40], [190, 90, 47], 
    [175, 64, 39], [177, 70, 40], [159, 55, 39], 
    [171, 75, 42], [181, 85, 43]]

#labels for our set of measurements
Y = ['male', 'female', 'female', 'female', 'male', 'male',
    'male', 'female', 'male', 'female', 'male']

#new set of measurements to be predicted
challenger = [[190, 70, 43]]

#using tree to classify 'challenger' based on best fit from our X,Y sets and make a prediction
tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(X, Y)
prediciton1 = tree_clf.predict(challenger)
print (prediciton1)

#
svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_clf = svc_clf.fit(X, Y)
prediction2 = svc_clf.predict(challenger)
print (prediction2)


# julio = tree_clf.predict([[180, 95, 44]])
# print ("\nJulio is a ")
# print (julio)

# HaX = clf.predict([[182, 70, 44]])
# print ("\nHaX is a")
# print (HaX)