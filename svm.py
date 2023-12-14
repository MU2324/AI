#SVM

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset as an example
iris = datasets.load_digits()
X = iris.data
y = iris.target

print(X)
print("========================")
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
cls = svm.SVC(kernel="linear")
X_train,y_train =iris.data[:-10],iris.target[:-10]

# Train the classifier on the training data
cls.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = cls.predict(X_test)

print(cls.predict(iris.data[:-10]))

# Evaluate the accuracy of the classifier

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

plt.imshow(iris.images[9], interpolation='nearest')
plt.show()

