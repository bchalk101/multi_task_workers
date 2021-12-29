import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size=64):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

class FirstPart(tf.keras.Model):
  def __init__(self) -> None:
      super().__init__()
      self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu')
  
  def call(self, input):
      out = tf.keras.layers.InputLayer(input_shape=(28, 28))(input)
      out = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(out)
      return self.conv2d(out)

class MModel(tf.keras.Model):
  def __init__(self) -> None:
      super().__init__()
      self.dense1 = tf.keras.layers.Dense(128, activation='relu')
      self.dense2 = tf.keras.layers.Dense(10)

  def call(self, input):
      out = tf.keras.layers.Flatten()(input)
      out = self.dense1(out)
      return self.dense2(out)

class ConnectedModel(tf.keras.Model):
  def __init__(self) -> None:
      super().__init__()
      self.first_model = FirstPart()
      self.second_model = MModel()

  def call(self, input):
    out = self.first_model(input)
    return self.second_model(out)
