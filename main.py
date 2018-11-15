# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import model
import preprocessing

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path',
                    '/home/tianming/data/' +
                    'AgriculturalDisease_trainingset/train.record',
                    'Path to training tfrecord file.')
flags.DEFINE_string('val_record_path',
                    '/home/tianming/data/' +
                    'AgriculturalDisease_trainingset/validation.record',
                    'Path to training tfrecord file.')
flags.DEFINE_string('checkpoint_path',
                    '/data2/raycloud/model_zoo/' +
                    'resnet_v1_152.ckpt',
                    'Path to pretrained ResNet model.')
flags.DEFINE_string('logdir', './training', 'Path to log directory.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')#0.001
flags.DEFINE_float(
    'learning_rate_decay_factor', 0.7, 'Learning rate decay factor.')
flags.DEFINE_float(
    'num_epochs_per_decay', 3.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
flags.DEFINE_integer('num_samples', 31718, 'Number of samples.')
flags.DEFINE_integer('num_steps', 40000, 'Number of steps.')
flags.DEFINE_integer('batch_size', 22, 'Batch size') #32

FLAGS = flags.FLAGS


def get_record_dataset(record_path,
                       reader=None,
                       num_samples=50000,
                       num_classes=7):
    """Get a tensorflow record file.

    Args:

    """
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
                                                                     dtype=tf.int64))}

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                              format_key='image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py

    Args:
        num_samples_per_epoch: he number of samples in each epoch of training.
        global_step: The global_step tensor.

    Returns:
        A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')


def get_init_fn():
    """Returns a function run by che chief worker to warm-start the training.

    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py

    Note that the init_fn is only run when initializing the model during the
    very first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.logdir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists ' +
            'in %s' % FLAGS.logdir)
        return None

    if tf.gfile.IsDirectory(FLAGS.logdir):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    variables_to_restore = slim.get_variables_to_restore()
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)

def get_trainable_variables(trainable_scopes):
    """Returns a list of variables to train.
    Returns:
        A list of variables to train by the optimizer.
    """

    if trainable_scopes is None:
        return tf.trainable_variables()

    trainable_scopes = [scope.strip() for scope in trainable_scopes]

    variables_to_train = []
    for scope in trainable_scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    num_samples = FLAGS.num_samples
    dataset = get_record_dataset(FLAGS.record_path, num_samples=num_samples,
                                 num_classes=61)
    val_dataset = get_record_dataset(FLAGS.val_record_path, num_samples=num_samples,
                                 num_classes=61)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    val_data_provider = slim.dataset_data_provider.DatasetDataProvider(val_dataset)
    image, label = data_provider.get(['image', 'label'])
    val_image, val_label = val_data_provider.get(['image', 'label'])
    img_size=368
    # Border expand and resize
    image = preprocessing.border_expand(image, resize=True, output_height=img_size,
                                        output_width=img_size)
    # val_image = preprocessing.border_expand(val_image, resize=True, output_height=img_size,
    #                                     output_width=img_size)

    inputs, labels = tf.train.batch([image, label],
                                    batch_size=FLAGS.batch_size,
                                    # capacity=5*FLAGS.batch_size,
                                    allow_smaller_final_batch=True)
    # val_inputs, val_labels = tf.train.batch([val_image, val_label],
    #                                 batch_size=FLAGS.batch_size,
    #                                 # capacity=5*FLAGS.batch_size,
    #                                 allow_smaller_final_batch=True)

    cls_model = model.Model(is_training=True, num_classes=61)


    preprocessed_inputs = cls_model.preprocess(inputs)


    prediction_dict = cls_model.predict(preprocessed_inputs)


    loss_dict = cls_model.loss(prediction_dict, labels)

    loss = loss_dict['loss']

    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, labels)
    tf.summary.scalar('loss', loss)

    tf.summary.scalar('accuracy', acc)

    global_step = slim.create_global_step()
    learning_rate = configure_learning_rate(num_samples, global_step)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)
    #    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    # trainable_vars = get_trainable_variables(['block1','block2','block4','pool5','conv1','pool1','pool5','prePredict2','Predict'])
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             # variables_to_train=trainable_vars,
                                             summarize_gradients=True)
    tf.summary.scalar('learning_rate', learning_rate)

    init_fn = get_init_fn()
    saver= tf.train.Saver(max_to_keep=40)
    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        init_fn=init_fn, number_of_steps=FLAGS.num_steps,
                        save_summaries_secs=30,
                        save_interval_secs=600,saver=saver)


if __name__ == '__main__':
    tf.app.run()

