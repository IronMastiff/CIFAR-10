'Input CIFAR-10'

import os

import tensorflow.python.platform
import tensorflow as tf

from tensorflow.python.platform import gfile

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def read_cifar10( filename_queue ):
    class CIFAR10Record( object ):
        pass
    result = CIFAR10Record
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader( record_bytes = record_bytes )
    result.key, value = reader.read( filename_queue )

    record_bytes = tf.decode_raw( value, tf.uint8 )
    result.label = tf.cast( tf.slice( record_bytes, [0], [label_bytes] ), tf.int32 )
    depth_major = tf.reshape( tf.slice( record_bytes, [label_bytes], [image_bytes] ), [result.depth,
                                                                                       result.height, result.width] )
    result.uint8iamge = tf.transpose( depth_major, [1, 2, 0] )
    return result

def _generate_iamge_and_label_batch( image, label, min_queue_examples, batch_size ):
    num_preprocess_threads = 16
    images, label_batch = tf.train.suffle_batch(    #tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue) 将队列中的数据打乱后取出
        [image, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_duqueue = min_queue_examples
    )
    tf.image_summary( 'iamges', images )           #image_summary tf的图像数据可视化

    return images, tf.reshape( label_batch, [batch_size] )

def distorted_inputs( data_dir, batch_size ):                          #用于训练
    filenames = [os.path.join( data_dir, 'data_batch_%d.bin' % i ) for i in xrange( 1, 6 )]
    for f in filenames:
        if not gfile.Exists( f ):
            raise ValueError( 'Falle to find file:' + f )
    filename_queue = tf.train.string_input_producer( filenames )

    read_input = read_cifar10( filename_queue )
    reshaped_image = tf.cast( read_input.unit8image, tf.float32 )

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_iamge = tf.image.random_crop( reshaped_image, [height, width] )             #随机裁剪

    distorted_image = tf.image.random_flip_left_right( distorted_iamge )

    distoretd_iamge = tf.image.random_brightness( distorted_iamge, max_delta = 63 )

    distorted_iamge = tf.image.random_contrast( distorted_iamge, lower = 0.2, upper = 1.8 )

    float_iamge = tf.image.per_iamge_whitening( distorted_iamge )

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int( NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue )
    print( 'Filling queue with %d CIFAR images before starting to train.''This will take a few minutes.' %min_queue_examples )
    return _generate_iamge_and_label_batch( float_iamge, read_input.label, min_queue_examples, batch_size )

def inputs( eval_data, data_dir, batch_size ):                              #用于测试
    if not eval_data:
        filenames = [os.path.join( data_dir, 'data_batch_%d.bin' % i ) for i in xrange( 1, 6 )]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join( data_dir, 'test_batch.bin' )]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not gfile.Exists( f ):
            raise ValueError( 'Fail to find file:' + f )

    filename_queue = tf.train.string_input_producer( filenames )

    read_input = read_cifar10( filename_queue )
    reshaped_iamge = tf.cast( read_input.unit8image, tf.float32 )

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_iamge = tf.iamge.resize_iamge_with_crop_or_pad( reshaped_iamge, width, height )       #中央裁剪
    float_iamge = tf.iamge.per_image_whitening( resized_iamge )
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int( num_examples_per_epoch * min_fraction_of_examples_in_queue )
    return _generate_iamge_and_label_batch( float_iamge, read_input.label, min_queue_examples, batch_size )















