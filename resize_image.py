"""Resizes all images from a source_dir to a target_dir.

This program reads all images from a folder, --source_dir, resize them
based on the --img_width and --img_height arguments, and save them to
--target_dir.

The --source_dir argument sets the folder where the images to be resized are located.

The --target_dir argument sets the folder where the resized images will be saved.

The --img_width argument sets the new image width.

The --img_height argument sets the new image height.

If source_dir is the same as target_dir the original photos will be overwritten by the resized ones.
"""
import argparse
import os
from os.path import basename
import glob
import scipy.misc

FLAGS = None

def resize_images(src_dir, des_dir, img_w, img_h):
    '''Reads all images from src_dir, resizes them based on img_w and img_h and saves them to 
des_dir'''
    # read each jpg file in src_dir
    for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
        # gets the file base name
        photo = basename(filename)
        print 'Resizing ' + filename + ' to ' + os.path.join(des_dir, photo)
        # reads the iamge data
        array_image = scipy.misc.imread(filename)
        # resize the image
        array_resized_image = scipy.misc.imresize(array_image, (img_w, img_h), interp='nearest', mode=None)
        # saves the resized image to the des_dir with the same base name
        scipy.misc.imsave(os.path.join(des_dir, photo), array_resized_image)

def main():
    '''Main function'''
    src_dir = FLAGS.source_dir
    des_dir = FLAGS.target_dir
    img_w = FLAGS.img_width
    img_h = FLAGS.img_height
    resize_images(src_dir, des_dir, img_w, img_h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        default='./',
        help="""\
        The source directory with images to be resized\
        """
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        default='./',
        help="""\
        The target directory where images will be saved to\
        """
    )
    parser.add_argument(
        '--img_width',
        type=int,
        required=True,
        default=28,
        help="""\
        The new image width\
        """
    )
    parser.add_argument(
        '--img_height',
        type=int,
        required=True,
        default=28,
        help="""\
        The new image height\
        """
    )
    FLAGS, _ = parser.parse_known_args()
    main()
