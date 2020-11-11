#!/usr/bin/python3

import tensorflow as tf;

def ResNet101(input_shape):

  inputs = tf.keras.Input(input_shape);
  results = tf.keras.applications.ResNet101(weights = 'imagenet', include_top = False)(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet101');

def ImprintedWeights(channel, nclasses, feature_extractor_model = None, fc_model = None):

  inputs = tf.keras.Input((None, None, channel));
  resnet101 = ResNet101(inputs.shape[1:]);
  # load pretrain model
  if feature_extractor_model is not None:
    resnet101.load_weights(feature_extractor_model);
  results = resnet101(inputs);
  results = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), method = tf.image.ResizeMethod.BILINEAR))([results, inputs]);
  results = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4), name = 'prototype')(results);
  fc = tf.keras.layers.Conv2D(nclasses, kernel_size = (1,1), padding = 'same', activation = tf.keras.activations.softmax, name = 'full_conv');
  if fc_model is not None:
    fc.load_weights(fc_model);
  results = fc(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  iw = ImprintedWeights(channel = 3, nclasses = 10);
  import numpy as np;
  inputs = np.random.normal(size = (8,256,256,3));
  results = iw(inputs);
  iw.save('iw.h5');
