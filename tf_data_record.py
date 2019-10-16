import os

import tensorflow as tf
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from tensorflow.core.example.example_pb2 import Example

data_dir = './DOG_DATA'

classes = ['husky', 'jiwawa', 'poodle', 'qiutian']


def write_tfrecord():
    writer = tf.python_io.TFRecordWriter(path='./TFData/dog.tfrecords')
    for i, name in enumerate(classes):
        class_path = os.path.join(data_dir, name)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img: JpegImageFile = Image.open(img_path)
            img = img.resize((64, 64))
            data = img.tobytes(encoder_name='raw')
            one_feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
            }

            features = tf.train.Features(feature=one_feature)

            example: Example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord():
    input_producer = tf.train.string_input_producer(['./TFData/dog.tfrecords'])
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(input_producer)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'img_raw': tf.FixedLenFeature(shape=[], dtype=tf.string)
    })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label', tf.int64])

    return image, label

