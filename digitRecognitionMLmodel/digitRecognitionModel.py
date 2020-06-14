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
inp_cv, inp_test, out_cv, out_test = train_test_split(inp_test, out_test, train_size=0.3)

# scale input data to a value between 0 and 1
inp_train = inp_train / 255.0
inp_test = inp_test/255.0
inp_cv = inp_cv/255.0

output_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
inp_train = tf.reshape(inp_train, (len(inp_train), 28, 28, 1))
inp_test = tf.reshape(inp_test, (len(inp_test), 28, 28, 1))
inp_cv = tf.reshape(inp_cv, (len(inp_cv), 28, 28, 1))
print(inp_train.shape)
print(inp_test.shape)
print(inp_cv.shape)

def visualize_examples(num_examples, inp_dataset, out_dataset, class_names):
    #Takes in a number of examples to visualize, the input dataset, and the label dataset and displays a plot of the examples
    inp_dataset = tf.reshape(inp_dataset, (len(inp_train), 28, 28))
    if int(math.sqrt(num_examples)) != math.sqrt(num_examples):
        raise Exception('Can Only Plot A Perfect Square number of examples')
    plt.figure(figsize=(5, 5))
    for i in range(num_examples):
        plt.subplot(math.sqrt(num_examples), math.sqrt(num_examples), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(inp_dataset[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[out_dataset[i]])
    plt.show()

def create_model(numConvNeurons, numRegNeurons):
    augmentedDataset = ImageDataGenerator(
    zoom_range=0.15,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
    augmentedDataset.fit(inp_train)
    model = keras.Sequential([
        keras.layers.Conv2D(filters = numConvNeurons, kernel_size = (3, 3), strides = (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=100, kernel_size=(3, 3), strides=(3, 3), activation='relu',
                            input_shape=(28, 28, 1)),
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(numRegNeurons, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    # train model parameters with training data
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(augmentedDataset.flow(inp_train, out_train), validation_data=(inp_cv, out_cv), epochs = 7)
    return model

def evaluate_and_convert_model(numConvNeurons, numRegNeurons):
    # flatten image array, build model with arbitrary number of hidden units and 10 output units (for numbers between 0 and 9)
    model = create_model(numConvNeurons, numRegNeurons)

    # evaluate model accuracy on test data
    finalLoss, finalAccuracy = model.evaluate(inp_test, out_test, verbose=1)

    # display results
    print("accuracy on test data is ", finalAccuracy)

    model.save('finalModel.h5', save_format="h5")

    # get input, output node names for the TF graph from the Keras model
    input_name = model.inputs[0].name.split(':')[0]
    keras_output_node_name = model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]

    print(model.outputs)

    converted_model = tfcoreml.convert(tf_model_path='finalModel.h5',
                             image_input_names=input_name,
                             input_name_shape_dict={input_name: (1, 28, 28, 1)},
                             output_feature_names=[graph_output_node_name],
                             minimum_ios_deployment_target='13', image_scale=1.0/255.0)

    converted_model.author = 'Sonia'
    converted_model.short_description = 'Handwritten Digit Recognition with MNIST dataset'
    converted_model.save('myDigitRecognitionModel.mlmodel')

if __name__ == "__main__":

    #visualize_examples(25, inp_train, out_train, output_labels)

    evaluate_and_convert_model(numConvNeurons = 250, numRegNeurons = 450)