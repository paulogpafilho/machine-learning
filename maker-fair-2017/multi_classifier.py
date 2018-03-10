'''This program uses Google's Tensor Flow (https://www.tensorflow.org/)
Convolutional Neural Network (CNN) to train a model and classify images.
This code was based on Google's Tensor Flow tutorial on TF Layers:
https://www.tensorflow.org/tutorials/layers.
It can run in three different modes:
    - train: train a model
    - eval: evaluates the model accuracy
    - infer: infers an image based on the trained model
It takes the following arguments:
--mode: Required. Mode to run the program. Valid values are: 'train', 'eval', 'infer'.
--classes: Required. Number of classes represented in the training set.
--data_set: Required if --mode is train or eval. Data set for training and evaluating modes.
--img_file: Required if --mode is infer. Image to be inferred.
--steps: Optional. Defaults to 500. The number of training steps to train the model.
--model_dir: Optional. Defaults to the location where the program is running. Directory where 
            model and check points are (or going to be) saved to. If using a already trained model
            it will be used to evaluate and infer modes.
--img_w: Optional. Defaults to 28. Image pixels width used in the input data.
--img_h: Optional. Defaulst to 28. Image pixels height used in the input data.
--img_channels: Optional. Defaults to 3 (RGB). Image color channels from the input data.
Examples:
- To train: python multi_classifier.py --mode train --classes 7 --steps 100 --model_dir
./tmp/my_first_model --data_set ./sample.pkl
- To evaluate: python multi_classifier.py --mode eval --classes 7 --model_dir
./tmp/my_first_model --data_set ./sample.pkl
- To infer: python multi_classifier.py --mode infer --classes 7 --model_dir
./tmp/my_first_model --img_file ./original/apple_3695899.jpg
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import argparse
import pickle
import scipy
import glob
import os
from os.path import basename
import numpy as np
import tensorflow as tf

FLAGS = None
CLASSES = {"jobs":0, "newton":1}

# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.FATAL)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, FLAGS.img_w, FLAGS.img_h, FLAGS.img_channels])
    print("input_layer:")
    print(input_layer)

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("conv1:")
    print(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print("pool1:")
    print(pool1)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("conv2:")
    print(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("pool2:")
    print(pool2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
    print("pool2_flat:")
    print(pool2_flat)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=FLAGS.classes)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=FLAGS.classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def load_image(image):
    '''Loads and rescales an image for CNN infer mode. The picture is loaded from the
    parameter --img_file'''
    # Load the image file
    img = scipy.ndimage.imread(image, mode="RGB")
    # Scale it to IMAGE_W x IMAGE_H
    return scipy.misc.imresize(img, (FLAGS.img_w, FLAGS.img_h),
                               interp="bicubic").astype(np.float32, casting='unsafe')

def main(unused_argv):
    '''Main function. Processes the three modes available: train, eval and infer'''
    execution_start = datetime.datetime.now()
    print("Starting Execution at " + str(execution_start))
    print("Running Multi-Class Classification with flag: " + FLAGS.mode)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

    # Prediction
    if FLAGS.mode == tf.estimator.ModeKeys.PREDICT:
        predict_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": load_image(FLAGS.img_file)},
            y=None,
            num_epochs=1,
            shuffle=False)
        predict_results = mnist_classifier.predict(input_fn=predict_input, predict_keys=None,
                                                   hooks=None)
        print(list(predict_results))

    # Evaluate
    if FLAGS.mode == tf.estimator.ModeKeys.EVAL:
        train_data, train_labels, eval_data, eval_labels = pickle.load(open(FLAGS.data_set, "rb"))
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
    if FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
        # Load training and eval data
        train_data, train_labels, eval_data, eval_labels = pickle.load(open(FLAGS.data_set, "rb"))
        print("train_data: " + str(train_data.shape))
        print("train_labels: " + str(train_labels.shape))

        # Set up logging for training
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        # # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=train_data.shape[0],
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=FLAGS.steps,
            hooks=[logging_hook])

    # Test
    if FLAGS.mode == 'test':
        correct = 0
        wrong = 0
        for filename in glob.glob(os.path.join('./images/test', '*.jpg')):
            photo = basename(filename)
            print("photo: " + photo)
            predict_input = tf.estimator.inputs.numpy_input_fn(x={"x": load_image(filename)},y=None,num_epochs=1, shuffle=False)
            predict_results = mnist_classifier.predict(input_fn=predict_input, predict_keys=None, hooks=None)
            image_class = CLASSES[basename(photo.split("_")[0])]
            prediction = list(predict_results)[0]['classes']
            if image_class == prediction:
                correct += 1
            else:
                wrong += 1
        print("correct results: " + str(correct))
        print("wrong results: " + str(wrong))
            
    
    execution_end = datetime.datetime.now()
    print("Execution Finished at: " + str(execution_end))
    print("Total Time: " + str(execution_end - execution_start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval', 'infer', 'test'],
        help="""\
        Mode to run the script.\
        Train - trains a new model based on the provided --data_set.\
        Eval - evaluates the trained model accurancy based on the provided --data_set.\
        Infer - infer an image, --img_file, based on the saved model.\
        """
    )
    parser.add_argument(
        '--classes',
        type=int,
        required=True,
        help="""\
        Number of classes represented in the training and eval data set.\
        """
    )
    parser.add_argument(
        '--img_w',
        type=int,
        default=28,
        required=False,
        help="""\
        Image width from the input data.\
        """
    )
    parser.add_argument(
        '--img_h',
        type=int,
        default=28,
        required=False,
        help="""\
        Image height from the input data.\
        """
    )
    parser.add_argument(
        '--img_channels',
        type=int,
        default=3,
        required=False,
        help="""\
        Image color channels from the input data.\
        """
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        required=False,
        help="""\
        Number of training steps to train the model. If not provided defaults to 500\
        """
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./',
        required=False,
        help="""\
        Directory where models and check points are (or going to be) saved.\
        """
    )
    parser.add_argument(
        '--img_file',
        type=str,
        required=False,
        help="""\
        Image to be inferred. Required if --mode is infer\
        """
    )
    parser.add_argument(
        '--data_set',
        type=str,
        required=False,
        help="""\
        Images data set for training and evaluating. Required if --mode is train or eval.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.mode == tf.estimator.ModeKeys.PREDICT:
        if FLAGS.img_file is None:
            parser.error("--img_file must be provided if --mode is infer.")
    if FLAGS.mode == tf.estimator.ModeKeys.TRAIN or FLAGS.mode == tf.estimator.ModeKeys.EVAL:
        if FLAGS.data_set is None:
            parser.error("--data_set must be provided if --mode is train or eval")
    tf.app.run()
