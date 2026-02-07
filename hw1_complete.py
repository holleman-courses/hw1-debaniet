#!/usr/bin/env python

# TensorFlow and tf.keras
from xml.parsers.expat import model
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
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
  ], name="model1")
  return model

def build_model2():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
  ], name="model2")

  return model

def build_model3():
  model = Sequential([
    layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),

    layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(10)
  ], name="model3")
  return model

def build_model50k():
  model = Sequential([

    layers.Conv2D(48, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(160, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.5),
    layers.Dense(10)

  ], name="model50k")
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
  (all_train_imgs, all_train_labels), (all_test_images, all_test_labels) = tf.keras.datasets.cifar10.load_data()
  
  all_train_imgs = all_train_imgs / 255.0
  all_test_images = all_test_images / 255.0

  val_split = int(0.2*len(all_train_imgs))
  
  val_imgs = all_train_imgs[:val_split]
  val_labels = all_train_labels[:val_split]

  train_imgs = all_train_imgs[val_split:]
  train_labels = all_train_labels[val_split:]

  # Build and train model 1
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


  # Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.summary() # print model summary to check number of parameters

  model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history2 = model2.fit(train_imgs, train_labels, epochs=15, validation_data=(val_imgs, val_labels))

  print("Model 2 training complete.")
  train_loss, train_acc = model2.evaluate(train_imgs, train_labels, verbose=0)
  val_loss, val_acc = model2.evaluate(val_imgs, val_labels, verbose=0)
  test_loss, test_acc = model2.evaluate(all_test_images, all_test_labels, verbose=0)

  print(f"Model 2 Train Accuracy: {train_acc:.4f}")
  print(f"Model 2 Val Accuracy: {val_acc:.4f}")
  print(f"Model 2 Test Accuracy: {test_acc:.4f}")
  
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
  
  filename = 'test_image_classname.jpg'
  
  try:
    img = keras.utils.load_img(filename, target_size=(32, 32))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model2.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f}% confidence.")
  except Exception as e:
    print(f"Could not load and predict on image {filename}. Error: {e}") 

  
  ## Repeat for model 3 and your best sub-50k params model
  
  # Build, compile, and train model 3 
  model3 = build_model3()
  model3.summary() # print model summary to check number of parameters

  model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history3 = model3.fit(train_imgs, train_labels, epochs=30, validation_data=(val_imgs, val_labels))

  print("Model 3 training complete.")
  train_loss, train_acc = model3.evaluate(train_imgs, train_labels, verbose=0)
  val_loss, val_acc = model3.evaluate(val_imgs, val_labels, verbose=0)
  test_loss, test_acc = model3.evaluate(all_test_images, all_test_labels, verbose=0)

  print(f"Model 3 Train Accuracy: {train_acc:.4f}")
  print(f"Model 3 Val Accuracy: {val_acc:.4f}")
  print(f"Model 3 Test Accuracy: {test_acc:.4f}")

  # Build, compile, and train model 50k 
  model50k = build_model50k()
  model50k.summary() # print model summary to check number of parameters
  
  model50k.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history50k = model50k.fit(train_imgs, train_labels, epochs=30, validation_data=(val_imgs, val_labels))

  print("Model 50k training complete.")
  train_loss, train_acc = model50k.evaluate(train_imgs, train_labels, verbose=0)
  val_loss, val_acc = model50k.evaluate(val_imgs, val_labels, verbose=0)
  test_loss, test_acc = model50k.evaluate(all_test_images, all_test_labels, verbose=0)

  print(f"Model 50k Train Accuracy: {train_acc:.4f}")
  print(f"Model 50k Val Accuracy: {val_acc:.4f}")
  print(f"Model 50k Test Accuracy: {test_acc:.4f}")