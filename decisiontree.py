#Decision Tree

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

clf = DecisionTreeClassifier()
model = clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy is ",accuracy)

fig = plt.figure(figsize=(15,10))
image = tree.plot_tree(clf, feature_names=iris.feature_names, class_names=class_names, filled=True)
plt.show()
