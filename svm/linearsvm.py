import numpy as np #for array conversions
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

features_train = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
labels_train = [0,1,0,1,0,1]
feature_test = [(0.58,0.76)]

clf = svm.SVC(kernel = "linear")
clf.fit(features_train , labels_train)

print (clf.predict(feature_test))

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(features_train[:, 0], features_train[:, 1], c = labels_train)
plt.legend()
plt.show()
