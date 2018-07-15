import numpy as np #for array conversions
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

#Giving hardcoded value of the features
features_train = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])

#Deciding in which category they belong
labels_train = [0,1,0,1,0,1]

#Features to test the model
feature_test = [(0.58,0.76)]

#Using the support vector classifier as model
clf = svm.SVC(kernel = "linear")
#Training the model
clf.fit(features_train , labels_train)
#Predicting the value of the model
print (clf.predict(feature_test))


##Plotting the Line graph
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(features_train[:, 0], features_train[:, 1], c = labels_train)
plt.legend()
plt.show()
