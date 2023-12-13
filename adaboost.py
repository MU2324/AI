#Adaboost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1234)

clf = DecisionTreeClassifier(max_depth=1)
clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print('Accuracy of individual tree ',accuracy)

boost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3) , n_estimators=50 , random_state=1234)
boost_clf.fit(X_train , y_train)

y_adaboost_pred = boost_clf.predict(X_test)

accuracy = accuracy_score(y_test , y_adaboost_pred)
print('Accuracy after boosting ',accuracy)
