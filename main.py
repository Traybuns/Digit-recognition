import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
data.tf.keras.datasets.mnist
from tensorflow.python.keras.metrics import accuracy


(x_train,y_train), (x_test,y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.sequential()
model.add(tf.keras.layers.flatten(input_shape=(28,28)))
model.add(tf.keras.layers.dense(units-128), activation=tf.mn.relu)
model.add(tf.keras.layers.dense(units-128), activation=tf.mn.relu)
model.add(tf.keras.layers.dense(units=10,activation=tf.mn.softmax))
model.compile(optimizer= 'adam',loss='sparse_categorecal_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)   

loss,accuracy= model.evaluate(x_test,y_test)
print (accuracy)
print (loss)


