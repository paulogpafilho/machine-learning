'''This program uses Google's Tensor Flow (https://www.tensorflow.org/)
Convolutional Neural Network (CNN) to train a model and classify images.
This code was based on Google's Tensor Flow tutorial on TF Layers: 
https://www.tensorflow.org/tutorials/layers. 
Set --mode argument to train, eval or infer
Set --model_dir to the directory where models and check points are (or going to be) saved.
Set --img_file to the image to be inferred.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import scipy
import datetime
import argparse

FLAGS = None
IMAGE_W = 28
IMAGE_H = 28
IMAGE_COLOR_CHANNELS = 3
CLASSES = 7
TRAINING_STEPS = 2000
# MODEL_DIR="./tmp/fruits_convnet_model"
# IMAGE_FILE="./original/clementine_2209404.jpg"

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, IMAGE_W, IMAGE_H, IMAGE_COLOR_CHANNELS])

  # Convolutional Layer #1 and Pooling Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=CLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CLASSES)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def loadImage(image):
    # Load the image file
    img = scipy.ndimage.imread(image, mode="RGB")
    # Scale it to IMAGE_W x IMAGE_H
    return scipy.misc.imresize(img, (IMAGE_W, IMAGE_H), interp="bicubic").astype(np.float32, casting='unsafe')

def main(unused_argv):
  execution_start = datetime.datetime.now()
  print("Starting Execution at " + str(execution_start))
  print("Running Multi-Class Classification with flag: " + FLAGS.mode)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

  # Prediction
  if (FLAGS.mode == tf.estimator.ModeKeys.PREDICT):
    predict_input = tf.estimator.inputs.numpy_input_fn(
      x={"x": loadImage(FLAGS.img_file)},
      y=None,
      num_epochs=1,
      shuffle=False)
    predict_results = mnist_classifier.predict(input_fn=predict_input,predict_keys=None,hooks=None)
    print(list(predict_results))

  # Evaluate
  if (FLAGS.mode == tf.estimator.ModeKeys.EVAL):
    train_data, train_labels, eval_data, eval_labels  = pickle.load(open("sample.pkl", "rb"))
    print("eval_data: " + str(eval_data.shape))
    print("eval_labels: " + str(eval_labels.shape))
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

  # Train
  if (FLAGS.mode == tf.estimator.ModeKeys.TRAIN):
    # Load training and eval data
    train_data, train_labels, eval_data, eval_labels  = pickle.load(open("sample.pkl", "rb"))
    print("train_data: " + str(train_data.shape))
    print("train_labels: " + str(train_labels.shape))

    # Set up logging for training
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    mnist_classifier.train(
      input_fn=train_input_fn,
      steps=TRAINING_STEPS,
      hooks=[logging_hook])

  execution_end = datetime.datetime.now()
  print("Execution Finished at: " + str(execution_end))
  print("Total Time: " + str(execution_end - execution_start))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mode',
      type=str,
      default='infer',
      choices=['train','eval','infer'],
      help="""\
      Mode to run the script. Valid values are:
      train, eval and infer.\
      """
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='./',
      required=True,
      help="""\
      Directory where models and check points are (or going to be) saved.\
      """
  )
  parser.add_argument(
      '--img_file',
      type=str,
      help="""\
      Image to be inferred. Required if --mode is infer\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  if (FLAGS.mode == tf.estimator.ModeKeys.PREDICT) and (FLAGS.img_file is None):
      raise ValueError("--img_file must be provided if --mode is infer")
  tf.app.run()
