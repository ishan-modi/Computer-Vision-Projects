from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

#Scaling
x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

#Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Train
model.fit(x_train,y_cat_train,epochs=10)

#Predict and Evaluate
model.evaluate(x_test,y_cat_test)

predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))

