from __future__ import division, print_function, absolute_import
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse
import datetime
import tensorflow as tf
import os, glob
from os.path import basename

#model = None
#MODEL_FOLDER='./tmp/fruits_convnet_model/'
IMAGE_W=28
IMAGE_H=28
IMAGE_COLOR_CHANNELS=3
CLASSES=7
MODEL_DIR="./tmp/fruits_convnet_model"
MONITOR_DIR = None
classes_map = {"apple": 0, "avocado": 1, "clementine": 2, "empty": 3, "kiwifruit": 4, "lime": 5, "plum": 6}
inventory = {}
tmp_inventory = {}

def getClassName(n):
    for name, code in classes_map.iteritems():
        if code == n:
            return name

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    start_model_Load = datetime.datetime.now()
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
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.jpg", "*.jpeg"]
    mnist_classifier = None

    def loadEstimator(self):
        # Create the Estimator
        a = datetime.datetime.now()
        print("Initiating Estimator...")
        self.mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)
        b = datetime.datetime.now()
        print("Estimator Initiated in: " + str(b-a))

    def processImage(self, image):
        time.sleep(.1) # allow some delay to file fully be copied into FS
        a = datetime.datetime.now()
        # Load the image file
        #img = scipy.ndimage.imread(args.image, mode="RGB")
        img = scipy.ndimage.imread(image, mode="RGB")
        # Scale it to 32x32
        img = scipy.misc.imresize(img, (IMAGE_W, IMAGE_H), interp="bicubic").astype(np.float32, casting='unsafe')
        # Predict
        predict_input = tf.estimator.inputs.numpy_input_fn(
          x={"x": img},
          y=None,
          num_epochs=1,
          shuffle=False)
        predict_results = self.mnist_classifier.predict(input_fn=predict_input,predict_keys=None,hooks=None)
        n = list(predict_results)[0]['classes']
        print("It's a(n) " + getClassName(n))
        b = datetime.datetime.now()
        print("Infer Time: " + str(b-a))

    def process(self, event):
        print(event.src_path + " is being inferred...")
        self.processImage(event.src_path)

    # def on_modified(self, event):
    #     self.process(event)

    def on_created(self, event):
        print("on_created")
        self.process(event)

    #def __init__(self):
    #    print("Initiating MyHandler...")
    #    self.loadEstimator()
        #super(PatternMatchingEventHandler, self).__init__()

if __name__ == '__main__':
    # args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Infer images from a directory to be monitored')
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        default='./',
        help="""\
        The directory to monitor\
        """
    )
    args, unparsed = parser.parse_known_args()
    MONITOR_DIR = args.directory;
    handler = MyHandler()
    handler.loadEstimator()
    observer = Observer()
    observer.schedule(handler, path=MONITOR_DIR if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
