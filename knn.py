#KNN

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print('accuracy ',accuracy)
