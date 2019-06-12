"""
Dataset generator, intended to be used with the Tensorflow Dataset API in the form of a TFRecords file. Originally
constructed to feed inputs to an implementation of SSD, this class should be general enough to feed any model if
provided an appropriate encoding function for that model.

Author: 1st Lt Ian McQuaid
Date: 16 Nov 2018
"""

import tensorflow as tf


class DatasetGenerator(object):
    def __init__(self,
                 tfrecord_name,
                 num_channels,
                 augment=False,
                 shuffle=False,
                 batch_size=4,
                 num_threads=1,
                 buffer=30,
                 encoding_function=None):
        """
        Constructor for the data generator class. Takes as inputs many configuration choices, and returns a generator
        with those options set.

        :param tfrecord_name: the name of the TFRecord to be processed.
        :param num_channels: the number of channels in the TFRecord images.
        :param augment: whether or not to apply augmentation to the processing chain.
        :param shuffle: whether or not to shuffle the input buffer.
        :param batch_size: the number of examples in each batch produced.
        :param num_threads: the number of threads to use in processing input.
        :param buffer: the prefetch buffer size to use in processing.
        :param encoding_function: a custom encoding function to map from the raw image/bounding boxes to the desired
                                  format for one's specific network.
        """
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.max_boxes_per_image = 10
        self.encode_for_network = encoding_function
        self.augment = augment
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.buffer = buffer
        self.tfrecord_path = tfrecord_name

        self.dataset = self.build_pipeline(tfrecord_name,
                                           augment=augment,
                                           shuffle=shuffle,
                                           batch_size=batch_size,
                                           num_threads=num_threads,
                                           buffer=buffer)

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):
        # Create and return iterator
        return self.dataset.make_one_shot_iterator()

    def _decode_satnet(self, example_proto):
        """
        This is the first step of the generator/augmentation chain. Reading the raw file out of the TFRecord is fairly
        straight-forward, though does require some simple fixes. For instance, the number of bounding boxes needs to be
        padded to some upper bound so that the tensors are all of the same shape and can thus be batched.

        :param example_proto: Example from a TFRecord file
        :return: The raw image and padded bounding boxes corresponding to this TFRecord example.
        """
        # Define how to parse the example
        features = {
            "width": tf.FixedLenFeature([], dtype=tf.int64),
            "height": tf.FixedLenFeature([], dtype=tf.int64),
            "images_raw": tf.VarLenFeature(dtype=tf.string),
            "ymin": tf.VarLenFeature(tf.float32),
            "ymax": tf.VarLenFeature(tf.float32),
            "xmin": tf.VarLenFeature(tf.float32),
            "xmax": tf.VarLenFeature(tf.float32),
            "classes": tf.VarLenFeature(tf.int64),
            "filename": tf.VarLenFeature(tf.string)
        }

        # Parse the example
        features_parsed = tf.parse_single_example(serialized=example_proto, features=features)
        width = tf.cast(features_parsed['width'], tf.int32)
        height = tf.cast(features_parsed['height'], tf.int32)

        ymin = tf.cast(tf.sparse.to_dense(features_parsed['ymin']), tf.float32)
        ymax = tf.cast(tf.sparse.to_dense(features_parsed['ymax']), tf.float32)
        xmin = tf.cast(tf.sparse.to_dense(features_parsed['xmin']), tf.float32)
        xmax = tf.cast(tf.sparse.to_dense(features_parsed['xmax']), tf.float32)
        classes = tf.cast(tf.sparse.to_dense(features_parsed['classes']), tf.float32)
        bboxes = tf.stack([ymin, xmin, ymax, xmax, classes], axis=1)

        # Because images can differ in number of bounding boxes, we need to pad to the same size
        # Before: bboxes is N X 5 where N is the number of boxes in that image
        paddings = tf.constant([[0, self.max_boxes_per_image], [0, 0]])
        paddings = paddings - (tf.constant([[0, 1], [0, 0]]) * tf.shape(bboxes)[0])
        bboxes = tf.pad(bboxes, paddings, constant_values=0.0)

        images = tf.sparse.to_dense(features_parsed['images_raw'], default_value="")
        images = tf.decode_raw(images, tf.uint16)
        print(self.num_channels)
        images = tf.reshape(images, [height, width, self.num_channels])

        # Normalize the image pixels to have zero mean and unit variance
        images = tf.image.per_image_standardization(images)

        return images

    def _decode_mnist(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [28, 28, self.num_channels])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)

        return image

    def build_pipeline(self, tfrecord_path, augment, shuffle, batch_size, num_threads, buffer):
        """
        Reads in data from a TFRecord file, applies augmentation chain (if
        desired), shuffles and batches the data.
        Supports prefetching and multithreading, the intent being to pipeline
        the training process to lower latency.

        :param tfrecord_path:
        :param augment: whether to augment data or not.
        :param shuffle: whether to shuffle data in buffer or not.
        :param batch_size: Number of examples in each batch returned.
        :param num_threads: Number of parallel subprocesses to load data.
        :param buffer: Number of images to prefetch in buffer.
        :return: the next batch, to be provided when this generator is run (see
        run_generator())
        """

        # Create the TFRecord dataset
        data = tf.data.TFRecordDataset(tfrecord_path)

        # Parse the record into tensors.
        # data = data.map(self._decode_mnist, num_parallel_calls=num_threads).prefetch(buffer)
        data = data.map(self._decode_satnet, num_parallel_calls=num_threads).prefetch(buffer)

        # If augmentation is to be applied
        if augment:
            # The only pixel-wise mutation possible on single channel imagery
            data = data.map(_vary_contrast, num_parallel_calls=num_threads).prefetch(buffer)

            # Technically, we only need rotation and one flip to get all possible orientations. But they are all here
            # anyways because it makes me feel better.
            data = data.map(_flip_left_right, num_parallel_calls=num_threads).prefetch(buffer)
            data = data.map(_flip_up_down, num_parallel_calls=num_threads).prefetch(buffer)
            data = data.map(_rotate_random, num_parallel_calls=num_threads).prefetch(buffer)

            # 50/50 chance of performing some crop, which is then randomly determined
            # data = data.map(_crop_random, num_parallel_calls=num_threads).prefetch(buffer)

        # If we decide to force all images to the same size, this line can be used
        # data = data.map(_resize_data, num_parallel_calls=num_threads).prefetch(buffer)

        # If the destination network requires a special encoding, do that here
        if self.encode_for_network is not None:
            data = data.map(self.encode_for_network, num_parallel_calls=num_threads).prefetch(buffer)

        if shuffle:
            data = data.shuffle(buffer)

        # Repeat the data forever (i.e. as many epochs as we desire)
        data = data.repeat()

        # Batch the data
        data = data.batch(batch_size)

        # Return a reference to this data pipeline
        return data


def _vary_contrast(image, bboxs):
    """
    Randomly varies the pixel-wise contrast of the image. This is the only pixel-wise augmentation that can be performed
    on single-channel imagery. The bounding boxes are not changed in any way by this function.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """

    cond_contrast = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast,
                    lambda: tf.image.random_contrast(image, 0.2, 1.8),
                    lambda: tf.identity(image))
    return image, bboxs


def _crop_random(image, bboxs):
    """
    Randomly either applies the random crop function, or returns the input with no changes.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """

    cond_crop = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    image, bboxs = tf.cond(cond_crop,
                           lambda: _perform_crop(image, bboxs),
                           lambda: (tf.identity(image), tf.identity(bboxs)))
    return image, bboxs


def _perform_crop(image, bboxs, min_crop_size=(400, 400)):
    """
    Randomly crops the image. Crop positions are chosen randomly, as well as the size (provided it is above the
    requested minimum size). If any bounding boxes are only partially included in the crop, the crop is enlarged so
    that the bounding box falls totally within the crop.

    Note: this makes batching difficult, as image sizes will no longer be consistent. To use this function in the
    augmentation chain, resizing will be needed after cropping but before batching.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :param min_crop_size: the smallest dimensions that a crop can take
    :return: the image and bounding box tensor with the desired transformation applied
    """

    # Get the image shape tensor
    img_shape = tf.shape(image)

    # First come up with a desired crop size, from given mins to the whole image (maxval is exclusive)
    crop_width = tf.random_uniform([], minval=min_crop_size[1], maxval=img_shape[1] + 1, dtype=tf.int32)
    crop_height = tf.random_uniform([], minval=min_crop_size[0], maxval=img_shape[0] + 1, dtype=tf.int32)

    # Now come up with crop offsets
    offset_width = tf.random_uniform([], minval=0, maxval=img_shape[1] - crop_width, dtype=tf.int32)
    offset_height = tf.random_uniform([], minval=0, maxval=img_shape[0] - crop_height, dtype=tf.int32)

    # If we ever split a box with our crop, increase the crop size to include it. Positives are already too scarce
    # First convert bounding boxes to pixel coordinates (rather than percents)
    bbox_convert_tensor = tf.stack([img_shape[0], img_shape[1], img_shape[0], img_shape[1], 1], axis=0)
    bbox_coords = bboxs * tf.cast(bbox_convert_tensor, dtype=tf.float32)
    bbox_coords = tf.cast(bbox_coords, dtype=tf.int32)

    # Need to figure out which boxes are totally within the crop, and which ones are totally outside the crop
    bbox_x_in_crop = tf.logical_and(tf.greater(bbox_coords[:, 1], offset_width),
                                    tf.greater(offset_width + crop_width, bbox_coords[:, 3]))
    bbox_y_in_crop = tf.logical_and(tf.greater(bbox_coords[:, 0], offset_height),
                                    tf.greater(offset_height + crop_height, bbox_coords[:, 2]))
    bbox_in_crop = tf.logical_and(bbox_x_in_crop, bbox_y_in_crop)

    bbox_x_out_of_crop = tf.logical_or(tf.greater(bbox_coords[:, 1], offset_width + crop_width),
                                       tf.greater(offset_width, bbox_coords[:, 3]))
    bbox_y_out_of_crop = tf.logical_or(tf.greater(bbox_coords[:, 0], offset_height + crop_height),
                                       tf.greater(offset_height, bbox_coords[:, 2]))
    bbox_out_of_crop = tf.logical_or(bbox_x_out_of_crop, bbox_y_out_of_crop)

    # Boxes not at all in the crop should be changed to negatives
    classes = bboxs[:, 4] * tf.cast(tf.logical_not(bbox_out_of_crop), dtype=tf.float32)

    # The problematic boxes are the ones that aren't either totally in the crop or totally out of the crop
    bboxes_split_mask = tf.logical_not(tf.logical_or(bbox_in_crop, bbox_out_of_crop))
    bboxes_split = tf.boolean_mask(bbox_coords, bboxes_split_mask)
    min_x = tf.reduce_min(bboxes_split[:, 1])
    max_x = tf.reduce_max(bboxes_split[:, 3])
    min_y = tf.reduce_min(bboxes_split[:, 0])
    max_y = tf.reduce_max(bboxes_split[:, 2])

    offset_width = tf.minimum(offset_width, min_x)
    offset_height = tf.minimum(offset_height, min_y)

    # A strange bug occurs when there were no split boxes. Have to make sure max_x and max_y are defined
    max_x = tf.maximum(max_x, offset_width)
    max_y = tf.maximum(max_y, offset_height)

    crop_height = tf.maximum(crop_height, max_y - offset_height)
    crop_width = tf.maximum(crop_width, max_x - offset_width)

    # The heavy lifting is done, time to make us a crop and transform our bounding boxes to the new coordinates
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)

    # Precision is an issue, need to cast to float before we start doing math here
    bbox_coords = tf.cast(bbox_coords, tf.float32)
    offset_width = tf.cast(offset_width, tf.float32)
    offset_height = tf.cast(offset_height, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)

    ymin = (bbox_coords[:, 0] - offset_height) / crop_height
    xmin = (bbox_coords[:, 1] - offset_width) / crop_width
    ymax = (bbox_coords[:, 2] - offset_height) / crop_height
    xmax = (bbox_coords[:, 3] - offset_width) / crop_width

    bbox_new = tf.stack([ymin, xmin, ymax, xmax, classes], axis=1)

    return image, bbox_new


def _flip_left_right(image, bboxs):
    """
    Randomly flips the image left or right, and transforms the bounding boxes appropriately.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """

    # Do the random image flip
    image_after = tf.image.random_flip_left_right(image)

    # Determine if a flip happened or not
    # Have to convert out of uint16...because tf apparently can't check equality of uint16...smh...
    image_one = tf.cast(image_after, dtype=tf.float32)
    image_two = tf.cast(image, dtype=tf.float32)

    # If every pixel were equal, this would be a NOT flip, hence the outer NOT to determine if this is a flip
    cond_flip = tf.logical_not(tf.reduce_all(tf.equal(image_one, image_two)))

    # This makes the computations a bit easier
    ymin = bboxs[:, 0]
    xmin = bboxs[:, 1]
    ymax = bboxs[:, 2]
    xmax = bboxs[:, 3]
    classes = bboxs[:, 4]

    # If we flipped, also flip the bounding boxes
    bboxs = tf.cond(cond_flip,
                    lambda: (tf.stack([ymin, 1 - xmax, ymax, 1 - xmin, classes], axis=1)),
                    lambda: tf.identity(bboxs))

    return image_after, bboxs


def _flip_up_down(image, bboxs):
    """
    Randomly flips the image up or down, and transforms the bounding boxes appropriately.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """

    # Do the random image flip
    image_after = tf.image.random_flip_up_down(image)

    # Determine if a flip happened or not
    # Have to convert out of uint16...because tf apparently can't check equality of uint16...smh...
    image_one = tf.cast(image_after, dtype=tf.float32)
    image_two = tf.cast(image, dtype=tf.float32)

    # If every pixel were equal, this would be a NOT flip, hence the outer NOT to determine if this is a flip
    cond_flip = tf.logical_not(tf.reduce_all(tf.equal(image_one, image_two)))

    # This makes the computations a bit easier
    ymin = bboxs[:, 0]
    xmin = bboxs[:, 1]
    ymax = bboxs[:, 2]
    xmax = bboxs[:, 3]
    classes = bboxs[:, 4]

    # If we flipped, also flip the bounding boxes
    bboxs = tf.cond(cond_flip,
                    lambda: (tf.stack([1 - ymax, xmin, 1 - ymin, xmax, classes], axis=1)),
                    lambda: tf.identity(bboxs))

    return image_after, bboxs


def _rotate_random(image, bboxs):
    """
    Randomly either applies the rotation function, or simply returns the input without any changes.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """
    cond_rotate = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    image, bboxs = tf.cond(cond_rotate,
                           lambda: _perform_rotation(image, bboxs),
                           lambda: (tf.identity(image), tf.identity(bboxs)))

    return image, bboxs


def _perform_rotation(image, bboxs):
    """
    Rotates images randomly either 90 degrees left or right. The decision to rotate or not is decided by _rotate_random.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :return: the image and bounding box tensor with the desired transformation applied
    """
    # Either rotate once clockwise or counter clockwise
    cond_direction = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    num_rotations = tf.cond(cond_direction,
                            lambda: 1,
                            lambda: -1)

    image = tf.image.rot90(image, k=num_rotations)

    # Rotate the bounding boxes with the image
    ymin = bboxs[:, 0]
    xmin = bboxs[:, 1]
    ymax = bboxs[:, 2]
    xmax = bboxs[:, 3]

    ymin, xmin, ymax, xmax = tf.cond(cond_direction,
                                     lambda: (1 - xmax, ymin, 1 - xmin, ymax),
                                     lambda: (xmin, 1 - ymax, xmax, 1 - ymin))

    bbox_new = tf.stack([ymin, xmin, ymax, xmax, bboxs[:, 4]], axis=1)
    return image, bbox_new


def _resize_data(image, bboxs, image_size=(2048, 2048)):
    """
    Resizes images to specified size. Intended to be applied as an element of the augmentation chain via the
    Tensorflow Dataset API map call.

    TODO: Currently this is not used, and is WRONG for use in SatNet. Resizing, if used, should be done via padding.

    :param image: input image tensor of Shape = Height X Width X Number of Channels
    :param bboxs: input bounding boxes of Shape = Number of Boxes X 4 (ymin, xmin, ymax, xmax)
    :param image_size: desired/output image size
    :return: the image and bounding box tensor with the desired transformation applied
    """

    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_images(image, image_size)
    image = tf.squeeze(image, axis=0)

    return image, bboxs
