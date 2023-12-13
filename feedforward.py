# FEED FORWARD BACK PRAPOGATION

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
y_binary = (y == 0).astype(int)

X_train , X_test , y_train , y_test = train_test_split(X , y_binary , test_size = 0.2)

model = tf.keras.Sequential([tf.keras.layers.Dense(8,input_dim = X_train.shape[1],activation = 'relu'),
                             tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer = 'adam' , loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train , y_train , epochs=50 , batch_size=32 , validation_split=0.2 , verbose = 0)

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test , y_pred_classes)
print('accuracy is ',accuracy)
