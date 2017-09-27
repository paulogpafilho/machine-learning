#!flask/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import request

import ast
import argparse
import os.path
import re
import sys
import tarfile
import datetime
import json
import glob
from os.path import basename
import scipy
import numpy as np
from six.moves import urllib
import tensorflow as tf

# source_folder = "./monitor"
# IMAGE_W=28
# IMAGE_H=28
# IMAGE_COLOR_CHANNELS=3
# CLASSES=7
# MODEL_DIR="./tmp/fruits_convnet_model"
# classes_map = {"apple": 0, "avocado": 1, "clementine": 2, "empty": 3, "kiwifruit": 4, "lime": 5, "plum": 6}
mnist_classifier = None
FLAGS = None

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def getClassName(n):
    for name, code in classes_map.iteritems():
        if code == n:
            return name

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    start_model_Load = datetime.datetime.now()
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, FLAGS.img_w, FLAGS.img_h, FLAGS.img_channels])

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

def loadEstimator():
    # Create the Estimator
    print("Initiating Estimator...")
    return tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

def processImage(image):
    a = datetime.datetime.now()
    # Load the image file
    #img = scipy.ndimage.imread(args.image, mode="RGB")
    img = scipy.ndimage.imread(os.path.join(FLAGS.monitor_directory, image), mode="RGB")
    # Scale it to 32x32
    img = scipy.misc.imresize(img, (FLAGS.img_w, FLAGS.img_h), interp="bicubic").astype(np.float32, casting='unsafe')
    # Predict
    predict_input = tf.estimator.inputs.numpy_input_fn(
      x={"x": img},
      y=None,
      num_epochs=1,
      shuffle=False)
    predict_results = mnist_classifier.predict(input_fn=predict_input,predict_keys=None,hooks=None)
    n = list(predict_results)[0]['classes']
    b = datetime.datetime.now()
    print("Infer Time: " + str(b-a))
    return getClassName(n)

def scan_directory():
    start_model_Load = datetime.datetime.now()
    ret = {}
    for filename in glob.glob(os.path.join(FLAGS.monitor_directory, '*.jpg')):
        print(basename(filename))
        ret[basename(filename.split(".jpg")[0])] = processImage(basename(filename))
    end_model_Load = datetime.datetime.now()
    print("All images classified in : " + str(end_model_Load - start_model_Load))
    return json.dumps(ret)

app = Flask(__name__)
#mnist_classifier = loadEstimator()

@app.route('/')
def index():
    return "Welcome to Smart Cabinet Service, please use the /inventory resource to get an updated cabinet inventory"

@app.route('/inventory')
def classify():
    if 'view' in request.args:
        if request.args.get('view') == 'consolidated':
            snapshot = ast.literal_eval(scan_directory())
            print(snapshot)
            ret = {}
            for e in snapshot:
                print(snapshot[e])
                if snapshot[e] in ret:
                    ret[snapshot[e]] += 1
                else:
                    ret[snapshot[e]] = 1
            print(str(ret))
        return str(str(ret))
    else:
        return scan_directory()

if __name__ == '__main__':
# source_folder = "./monitor"
# IMAGE_W=28
# IMAGE_H=28
# IMAGE_COLOR_CHANNELS=3
# CLASSES=7
# MODEL_DIR="./tmp/fruits_convnet_model"
# classes_map = {"apple": 0, "avocado": 1, "clementine": 2, "empty": 3, "kiwifruit": 4, "lime": 5, "plum": 6}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--monitor_directory',
        type=str,
        required=True,
        help="""\
        The directory where the pictures are stored.\
        Every time the endpoint is invoked, the server will read the folder\
        and infer all pictures it finds there. \
        """
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="""\
        The directory where the trained mode is. \
        """
    )
    parser.add_argument(
        '--img_w',
        type=int,
        default=28,
        help="""\
        Optional. Defaults to 28. Image width. The program will convert the image to the same width/height
        of the training model.\
        """
    )
    parser.add_argument(
        '--img_h',
        type=int,
        default=28,
        help="""\
        Optional. Defaults to 28. Image height. The program will convert the image to the same width/height
        of the training model.\
        """
    )
    parser.add_argument(
        '--img_channels',
        type=int,
        default=3,
        help="""\
        Optional. Defaults to 3. Image height. The program will convert the image to the same color channel
        of the training model.\
        """
    )
    parser.add_argument(
        '--classes_map_file',
        type=str,
        default='./sample_mapping.json',
        help="""\
        Optional. Defaults to ./sample_mapping.json. The class mapping JSON file\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    with open(str(FLAGS.classes_map_file), 'r') as myfile:
        data = myfile.read()
        classes_map = json.loads(data)
        CLASSES = len(classes_map)
    mnist_classifier = loadEstimator()
    app.run(debug=False,host='localhost',port=int(8080))
