# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a generator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

import os
import time
import argparse
import tensorflow as tf

# import data_provider
import networks
import util

from tensorflow.contrib.gan.python import losses as tfgan_losses


from dataset.dataset_generator import DatasetGenerator

tfgan = tf.contrib.gan

# tf.enable_eager_execution()


def _learning_rate(gan_type):
    # First is generator learning rate, second is discriminator learning rate.
    return {
        'unconditional': (1e-3, 1e-4),
        'conditional': (1e-5, 1e-4),
        'infogan': (0.001, 9e-5),
    }[gan_type]

# Two core changes in Keras
# Remove buggy error checks.FLAGS
def main(FLAGS):

    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):

            train_tfrecord_name = os.path.join(FLAGS.dataset_path,
                                               FLAGS.train_tfrecord)

            def cast_image_to_float(image, bbox=[]):

                return tf.cast(image, tf.float32)

            # Instantiate a wrapper .
            train_generator = DatasetGenerator(train_tfrecord_name,
                                               num_channels=FLAGS.num_channels,
                                               augment=FLAGS.augment_train_data,
                                               shuffle=FLAGS.shuffle_train_data,
                                               batch_size=FLAGS.batch_size,
                                               num_threads=FLAGS.num_dataset_threads,
                                               buffer=FLAGS.dataset_buffer_size,
                                               encoding_function=cast_image_to_float)

            train_iterator = train_generator.dataset.make_one_shot_iterator()

            images = train_iterator.get_next()

            print("\n\n\n\n\n")
            print(tf.shape(images))
            print("\n\n\n\n\n")
            # images, one_hot_labels, _ = data_provider.provide_data(
            #     'train', FLAGS.batch_size, FLAGS.dataset_dir, num_threads=4)

    # Define the GANModel tuple. Optionally, condition the GAN on the label or
    # use an InfoGAN to learn a latent representation.
    if FLAGS.gan_type == 'unconditional':
        gan_model = tfgan.gan_model(
            generator_fn=networks.unconditional_generator,
            discriminator_fn=networks.unconditional_discriminator,
            real_data=images,
            generator_inputs=tf.random_normal([FLAGS.batch_size,
                                               FLAGS.noise_dims]))

    elif FLAGS.gan_type == 'conditional':
        noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])
        gan_model = tfgan.gan_model(
            generator_fn=networks.conditional_generator,
            discriminator_fn=networks.conditional_discriminator,
            real_data=images,
            generator_inputs=(noise, one_hot_labels))

    elif FLAGS.gan_type == 'infogan':
        cat_dim, cont_dim = 10, 2
        generator_fn = functools.partial(
            networks.infogan_generator, categorical_dim=cat_dim)
        discriminator_fn = functools.partial(
            networks.infogan_discriminator, categorical_dim=cat_dim,
            continuous_dim=cont_dim)
        unstructured_inputs, structured_inputs = util.get_infogan_noise(
            FLAGS.batch_size, cat_dim, cont_dim, FLAGS.noise_dims)
        gan_model = tfgan.infogan_model(
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            real_data=images,
            unstructured_generator_inputs=unstructured_inputs,
            structured_generator_inputs=structured_inputs)

    # tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

    tf.summary.image('real_images',
                     images[:, :, :, :3],
                     max_outputs=2)

    # Get the GANLoss tuple. You can pass a custom function, use one of the
    # already-implemented losses from the losses library, or use the defaults.
    with tf.name_scope('loss'):
        mutual_information_penalty_weight = (1.0 if FLAGS.gan_type == 'infogan'
                                             else 0.0)
        gan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan_losses.minimax_generator_loss,
            discriminator_loss_fn=tfgan_losses.minimax_discriminator_loss,
            gradient_penalty_weight=1.0,
            mutual_information_penalty_weight=mutual_information_penalty_weight,
            add_summaries=True)
        tfgan.eval.add_regularization_loss_summaries(gan_model)

    # Get the GANTrain ops using custom optimizers.
    with tf.name_scope('train'):
        gen_lr, dis_lr = _learning_rate(FLAGS.gan_type)
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
            discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
            summarize_gradients=True,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

    if FLAGS.max_number_of_steps == 0:

        return

    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        logdir=FLAGS.train_log_dir,
        get_hooks_fn=tfgan.get_joint_train_hooks())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    # parser.add_argument('--dataset_path',
    #                     type=str,
    #                     default="C:\\research\\data\\mnist\\",
    #                     help='Path to the training TFRecord file.')

    # parser.add_argument('--train_tfrecord',
    #                     type=str,
    #                     default="train.tfrecords",
    #                     help='Name of the training TFRecord file.')

    # parser.add_argument('--valid_tfrecord',
    #                     type=str,
    #                     default="validation.tfrecords",
    #                     help='Name of the validation TFRecord file.')

    # parser.add_argument('--test_tfrecord',
    #                     type=str,
    #                     default="test.tfrecords",
    #                     help='Name of the testing TFRecord file.')

    # Set arguments and their default values
    parser.add_argument('--dataset_path', type=str,
                        default="C:\\research\\data\\satnet_single\\",
                        help='Path to the training TFRecord file.')

    parser.add_argument('--train_tfrecord', type=str,
                        default="single.tfrecords",
                        help='Name of the training TFRecord file.')

    parser.add_argument('--valid_tfrecord', type=str,
                        default="single.tfrecords",
                        help='Name of the validation TFRecord file.')

    parser.add_argument('--test_tfrecord', type=str,
                        default="single.tfrecords",
                        help='Name of the testing TFRecord file.')

    tb_file = os.path.join("./tensorboard/",
                           '{}_{}/'.format("gan", time.strftime('%m%d%I%M')))

    parser.add_argument('--train_log_dir',
                        type=str,
                        default=tb_file,
                        help='Directory where to write event logs.')

    parser.add_argument('--max_number_of_steps',
                        type=int,
                        default=1000000,
                        help='The maximum number of gradient steps.')

    parser.add_argument('--gan_type',
                        type=str,
                        default='unconditional',
                        help='Either `unconditional`, `conditional`, or `infogan`.')

    parser.add_argument('--grid_size',
                        type=int,
                        default=2,
                        help='Grid size for image visualization.')

    parser.add_argument('--noise_dims',
                        type=int,
                        default=64,
                        help='Dimensions of the generator noise vector.')

    parser.add_argument('--dataset_buffer_size',
                        type=int,
                        default=128,
                        help='Number of images to prefetch in the input pipeline.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=32,
                        help='Number of training epochs to run')

    parser.add_argument('--num_channels',
                        type=int,
                        default=1,
                        help='Number of channels in the input data.')

    parser.add_argument('--num_dataset_threads',
                        type=int,
                        default=64,
                        help='Number of threads to be used by the input pipeline.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch size to use in training, validation, and testing/inference.')

    parser.add_argument('--augment_train_data',
                        type=bool,
                        default=False,
                        help='If True, augment the training data')

    parser.add_argument('--shuffle_train_data',
                        type=bool,
                        default=True,
                        help='If True, shuffle the training data')

    FLAGS = parser.parse_args()

    # Launch training

    logging.set_verbosity(logging.INFO)
    # tf.app.run(FLAGS)

    main(FLAGS)
