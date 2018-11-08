# -*- coding: utf-8 -*-
import io
import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle


flags = tf.app.flags

common_path='/data2/raycloud/jingxiong_datasets/AIChanllenger/'
flags.DEFINE_string('images_dir',
                    common_path +
                    'AgriculturalDisease_trainingset/images',
                    'Path to images (directory).')
flags.DEFINE_string('annotation_path',
                    common_path +
                    'AgriculturalDisease_trainingset/' +
                    'AgriculturalDisease_train_annotations.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path',
                    '/home/tianming/data/AgriculturalDisease_trainingset/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_integer('resize_side_size', None, 'Resize images to fixed size.')

FLAGS = flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(image_path, label, resize_size=None):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    # Resize
    if resize_size is not None:
        image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()

    width, height = image.size

    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(encoded_jpg),
            'image/format': _bytes_feature('jpg'.encode()),
            'image/class/label': _int64_feature(label),
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width)}))
    return tf_example


def generate_tfrecord(annotation_dict, output_path, resize_size=None):
    num_valid_tf_example = 0
    print("output_path:"+output_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print('%s does not exist.' % image_path)
            continue
        tf_example = create_tf_example(image_path, label, resize_size)
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1

        if num_valid_tf_example % 100 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)


def provide(annotation_path=None, images_dir=None):
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')

    annotation_json = open(annotation_path, 'r')
    annotation_list = json.load(annotation_json)
    print(len(annotation_list))
    image_files = []
    annotation_dict = {}
    shuffle(annotation_list)
    for d in annotation_list:
        image_name = d.get('image_id')
        disease_class = d.get('disease_class')
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
        image_files.append(image_name)
        annotation_dict[image_name] = disease_class
    return image_files, annotation_dict

def show_label_cnt(annotation_dict):
    labels=[]
    for _, label in annotation_dict.items():
        labels.append(label)
    print(np.unique(labels))
    print(len(np.unique(labels)))


def main(_):
    images_dir = FLAGS.images_dir
    annotation_path = FLAGS.annotation_path
    record_path = FLAGS.output_path
    resize_size = FLAGS.resize_side_size
    print("annotation_path %s" % annotation_path)
    _, annotation_dict = provide(annotation_path, images_dir)
    generate_tfrecord(annotation_dict, record_path, resize_size)
    # show_label_cnt(annotation_dict)


if __name__ == '__main__':
    tf.app.run()
