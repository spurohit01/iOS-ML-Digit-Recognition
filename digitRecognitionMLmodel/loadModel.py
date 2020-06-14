import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft
import tfcoreml
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

handwrittenDigits = keras.datasets.mnist
(inp_train, out_train), (inp_test, out_test) = handwrittenDigits.load_data()
inp_train = inp_train / 255.0
myModel = tf.keras.models.load_model('finalModel.h5')
trainingExample = tf.reshape(inp_train[0], (1, 28, 28, 1))
print("Prediction is ", myModel.predict(trainingExample))