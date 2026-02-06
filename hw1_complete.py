#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(10)
  ], name="model1")
  return model

def build_model2():
  model = None # Add code to define model 1. This one should use depthwise separable convolutions.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model
# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
  (all_train_imgs, all_train_labels), (all_test_images, all_test_labels) = tf.keras.datasets.cifar10.load_data()
  val_split = int(0.2*len(all_train_imgs))
  
  val_imgs = all_train_imgs[:val_split]
  val_labels = all_train_labels[:val_split]

  train_imgs = all_train_imgs[val_split:]
  train_labels = all_train_labels[val_split:]

  ## Build and train model 1
  model1 = build_model1()
  model1.summary() # print model summary to check number of parameters

  model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history1 = model1.fit(train_imgs, train_labels, epochs=30, validation_data=(val_imgs, val_labels))

  print("Model 1 training complete.")
  train_loss, train_acc = model1.evaluate(train_imgs, train_labels, verbose=0)
  val_loss, val_acc = model1.evaluate(val_imgs, val_labels, verbose=0)
  test_loss, test_acc = model1.evaluate(all_test_images, all_test_labels, verbose=0)

  print(f"Model 1 Train Accuracy: {train_acc:.4f}")
  print(f"Model 1 Val Accuracy: {val_acc:.4f}")
  print(f"Model 1 Test Accuracy: {test_acc:.4f}")
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
