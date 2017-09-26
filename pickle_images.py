"""Image converting numpy arrays and serialized to pickle file.

This program reads all images from a folder, img_source_dir, and convert
them to numpy arrays.

It also creates labels for the images based on the image name (uses the
following format <CLASS_NAME>_<RANDOM STRING>.jpg) and maps them to the
classes_map_file, to obtain an integer for the desired class.

The classes_map_file is a dictionary that maps the classes names to integer
values for later processing by image recognition classifier.

The pickle file, data_file, will contain an array with 3/4 of the images as the training
data (X), an array with 3/4 of the correspondent labels (Y) for the training data,
an array with 1/4 of the images as test data (X_TEST) and one array with 1/4 of the labels
(Y_TEST) of the correspondent test data.

The --img_source_dir argument sets the directory where the images that
will be converted to numpy arrays are located.

The --classes_map_file argument sets the classes mapping, in a JSON format.
The JSON file must have the following format: {CLASS_NAME: CLASS_ID}, where CLASS_NAME is
a string representing the name of the class and CLASS_ID an integer representing the code
of the class.

The --data_file argument sets the name of the resulting pickle file containing the training
set and test sets.
"""
import glob
import pickle
import argparse
import json
from random import shuffle
import os
from os.path import basename
import numpy as np
import scipy.misc

FLAGS = None
DATA_FILE = "sample.pkl"
def convert_images(img_dir, class_mapping, data_file):
    """Convert all the images from img_dir into numpy arrays, creates labels for the images based 
on the image name and classes mapping, class_mapping, and saves them into pickle file, data_file"""
    print 'Data File: ' + str(data_file)
    # the training data
    X = []
    # the taining labels
    Y = []
    # the images directory listing, filtered by jpg format
    fotos_list = glob.glob(os.path.join(img_dir, '') + "*.jpg")
    # we shuffle the list to randominze the images in the resulting array
    shuffle(fotos_list)
    # reads each image in the file list
    for filename in fotos_list:
        print "Processing " + filename
        # from the first part of the image name gets the image mapping value
        image_class = class_mapping[basename(filename.split("_")[0])]
        # reads the image as array
        array_image = scipy.misc.imread(filename)
        # converts to float32
        array_image = array_image.astype('float32')
        # appends the image data to the training data set X
        X.append(np.array(array_image))
        # appends the image label to the training data set Y
        Y.append(image_class)

    # gets 1/4 of the data as test data and labels
    test_data = int(len(X[0:])/4.0)
    X_TEST = X[:test_data]
    Y_TEST = Y[:test_data]
    # resizes the training data and label arrays with the remaining 3/4 of the data
    X = X[test_data:]
    Y = Y[test_data:]

    print "serializing image arrays to " + data_file
    # opens the pickle file
    f = open(data_file, "w")
    # dumps the training data, training labels, test data and test labels
    pickle.dump((np.array(X), np.array(Y), np.array(X_TEST), np.array(Y_TEST)), f)

    # reads the pickle file and prints the array shapes
    print "reading arrays from pickle file " + data_file
    f = open(data_file, "rb")
    A, B, C, D = pickle.load(f)

    print A.shape
    print B.shape
    print C.shape
    print D.shape

def main():
    '''Main function'''
    img_dir = FLAGS.img_source_dir
    class_mapping = None
    data_file = DATA_FILE
    if FLAGS.data_file is not None:
        data_file = FLAGS.data_file
    with open(str(FLAGS.classes_map_file), 'r') as myfile:
        data = myfile.read()
        class_mapping = json.loads(data)
    print 'Data File: ' + data_file
    convert_images(img_dir, class_mapping, data_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_source_dir',
        type=str,
        required=True,
        default='./',
        help="""\
        The source directory to convert images from\
        """
    )
    parser.add_argument(
        '--classes_map_file',
        type=str,
        required=True,
        default='./sample_mapping.json',
        help="""\
        The class mapping JSON file\
        """
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=False,
        default='./sample.pkl',
        help="""\
        The resulting pickle file containing the training data set\
        """
    )
    FLAGS, _ = parser.parse_known_args()
    main()
