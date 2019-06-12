"""
An encoder for the SSD object detection model. Intended to be used via tensorflow's dataset API "map" call to convert
examples from a TFRecord file into the properly encoded target tensors for each head/predictor layer of SSD.

Author: 1st Lt Ian McQuaid
Date: 19 December 2018
"""
import numpy as np
import tensorflow as tf


class SSDEncoder(object):
    def __init__(self,
                 aspect_ratio_list=(1, 2, 3, 1 / 2, 1 / 3),
                 s_min=0.2,
                 s_max=0.9,
                 pred_layer_resolutions=([16, 16], [8, 8], [4, 4], [2, 2]),
                 labels=("Background", "Satellite"),
                 min_iou_to_match=0.5):
        """
        Constructor for the SSD encoder.

        :param aspect_ratio_list: aspect ratios desired in SSD's bounding boxes
        :param labels: class labels, to include the background class as labels[0]
        :param pred_layer_resolutions: resolution at each predictor layer. Implicitly the number of pred layers is
                                       given via the length of this argument
        :param s_min: the smallest scale of bounding box, in proportion of the original image. The smallest boxes
                      are used on the earliest/finest resolution predictor layers. Box scales are linearly spaced from
                      s_min to s_max
        :param s_max: the largest scale of bounding box, in proportion of the original image. The largest boxes
                      are used on the latest/coarsest resolution predictor layers. Box scales are linearly spaced from
                      s_min to s_max
        """
        self.labels = labels
        self.pred_layer_resolutions = pred_layer_resolutions
        self.num_pred_layers = len(pred_layer_resolutions)

        # Construct our anchor boxes
        # Layer 1 needs the smallest boxes, each additional layer will increase resolution by a factor of 2
        scales = np.linspace(s_min, s_max, self.num_pred_layers)
        self.anchors = [
            [(s * (ar ** (1 / 2)), s / (ar ** (1 / 2))) for ar in aspect_ratio_list] for s in scales
        ]
        self.min_iou_to_match = min_iou_to_match

    def encode_for_ssd(self, image, bboxes):
        """
        This is the function to be used by the Tensorflow Dataset API map function. This will transform the bounding
        boxes into the desired output tensors of each predictor layer. The image is used, but not changed by this
        process.

        Note that this is intended to be applied before the batching step, hence there not yet being a batch dimension.

        :param image: an input/example image from the input TFRecord file
                      Shape = Height X Width X Channels
        :param bboxes: a collection of bounding boxes for the input image. It is supposed that these are padded so that
                       each image has the same number of bounding boxes, thus logic is included to account for these
                       "negative" boxes. Shape = Number of Boxes X 5 (i.e. ymin, xmin, ymax, xmax, class)
        :return: the unaltered input image, and a tuple of each predictor layer's target tensor.
        """
        # First of all, we need to compute the ious for each ground truth/anchor box combination
        pred_layer_targets = []
        for layer_num in range(self.num_pred_layers):
            layer_res = self.pred_layer_resolutions[layer_num]
            layer_height = tf.cast(layer_res[0], dtype=tf.float32)
            layer_width = tf.cast(layer_res[1], dtype=tf.float32)

            # Get the absolute coordinates of the ground truth boxes in this predictor layer
            xmin = bboxes[:, 1]
            xmax = bboxes[:, 3]
            ymin = bboxes[:, 0]
            ymax = bboxes[:, 2]
            gt_labels = bboxes[:, 4]

            # Now expand these to have spatial dimensions as well
            xmin = tf.ones([layer_height, layer_width, tf.shape(xmin)[0]]) * xmin
            xmax = tf.ones([layer_height, layer_width, tf.shape(xmax)[0]]) * xmax
            ymin = tf.ones([layer_height, layer_width, tf.shape(ymin)[0]]) * ymin
            ymax = tf.ones([layer_height, layer_width, tf.shape(ymax)[0]]) * ymax
            gt_labels = tf.ones([layer_height, layer_width, tf.shape(gt_labels)[0]]) * gt_labels
            gt_boxes = [xmin, xmax, ymin, ymax]

            # Get and iterate over the anchor boxes for this predictor layer
            layer_anchors = self.anchors[layer_num]
            layer_ious_list = []
            layer_offset_list = []
            for anchor in layer_anchors:
                # Get the size of this anchor box
                anchor_width = anchor[0]
                anchor_height = anchor[1]

                # Form tensors of the positions of each grid in this layer
                anchor_x = tf.linspace(0., 1.0, tf.cast(layer_width, dtype=tf.int32))
                anchor_y = tf.linspace(0., 1.0, tf.cast(layer_height, dtype=tf.int32))
                anchor_x = anchor_x * tf.ones([layer_height, layer_width])
                anchor_y = anchor_y * tf.ones([layer_width, layer_height])
                anchor_y = tf.transpose(anchor_y, [1, 0])

                # Suppose that the anchor box was centered on this grid
                anchor_xmin = anchor_x - (anchor_width / 2)
                anchor_xmax = anchor_x + (anchor_width / 2)
                anchor_ymin = anchor_y - (anchor_height / 2)
                anchor_ymax = anchor_y + (anchor_height / 2)

                anchor_box_tensor = [anchor_xmin, anchor_xmax, anchor_ymin, anchor_ymax]

                # Compute the iou for each ground truth box with this anchor box
                # Shape = LH X LW X #Ground truth boxes
                anchor_iou_tensor = calc_iou(gt_boxes, anchor_box_tensor)

                # Some metrics for the ground truth boxes, for readabilities sake
                gt_width = xmax - xmin
                gt_height = ymax - ymin
                gt_center_x = (xmax + xmin) / tf.constant(2.0, dtype=tf.float32)
                gt_center_y = (ymax + ymin) / tf.constant(2.0, dtype=tf.float32)

                # The values that SSD predicts, as per the SSD paper
                pred_cx = (tf.transpose(gt_center_x, [2, 0, 1]) - anchor_x) / anchor_width
                pred_cy = (tf.transpose(gt_center_y, [2, 0, 1]) - anchor_y) / anchor_height
                pred_cx = tf.transpose(pred_cx, [1, 2, 0])
                pred_cy = tf.transpose(pred_cy, [1, 2, 0])
                pred_w = tf.log((gt_width + 1e-16) / anchor_width)
                pred_h = tf.log((gt_height + 1e-16) / anchor_height)

                # Put the values together into a single tensor per anchor box
                # Shape = LH X LW X #Ground truth boxes X 4
                pred_offsets = tf.stack([pred_cx, pred_cy, pred_w, pred_h], axis=-1)
                layer_ious_list.append(anchor_iou_tensor)
                layer_offset_list.append(pred_offsets)

            # Now concatenate into a single layer iou tensor
            # Shape = LH X LW X #Anchor boxes X #Ground truth boxes
            layer_iou_tensor = tf.stack(layer_ious_list, axis=2)

            # Now concatenate into a single layer offset tensor
            # Shape = LH X LW X #Anchor boxes X #Ground truth boxes X 4
            layer_offset_tensor = tf.stack(layer_offset_list, axis=2)

            # Match boxes where the iou > 0.5, or at least the best match you could find
            greater_match_tensor = tf.greater(layer_iou_tensor, self.min_iou_to_match)
            best_iou = tf.reduce_max(layer_iou_tensor, axis=[0, 1, 2])
            best_iou_tensor = tf.equal(layer_iou_tensor, best_iou)

            # An unexpected "gotcha": above we matched to padded boxes as well. Get rid of them.
            best_iou_tensor = tf.logical_and(best_iou_tensor, tf.cast(bboxes[:, 4], dtype=tf.bool))
            greater_match_tensor = tf.logical_and(greater_match_tensor, tf.cast(bboxes[:, 4], dtype=tf.bool))
            match_tensor = tf.logical_or(greater_match_tensor, best_iou_tensor)

            # Need to transpose the match tensor and cast to float32 so we can broadcast labels over it
            # Shape = #Anchor boxes X LH X LW X #Ground truth boxes
            match_tensor = tf.cast(tf.transpose(match_tensor, [2, 0, 1, 3]), dtype=tf.float32)

            # Remove unmatched pairs from the tensor and transpose back to normal
            # Shape = LH X LW X #Anchor boxes X #Ground truth boxes
            class_tensor = tf.cast(tf.transpose(match_tensor * gt_labels, [1, 2, 0, 3]), dtype=tf.int32)

            # One-hot encode these classes
            # Shape = LH X LW X #Anchor boxes X #Ground truth boxes X #Labels
            class_tensor = tf.cast(tf.one_hot(class_tensor, depth=len(self.labels)), dtype=tf.bool)

            # Collapse the ground truth dimension. If there are multiple objects in a given cell of a different class,
            # then they will both be noted in the class vector. Objects of the same class are considered to be the same.
            positive_class_tensor = tf.reduce_any(class_tensor[:, :, :, :, 1:], axis=3)
            background_class_tensor = tf.logical_not(tf.reduce_any(positive_class_tensor, axis=3))
            background_class_tensor = tf.expand_dims(background_class_tensor, axis=-1)
            positive_class_tensor = tf.cast(positive_class_tensor, dtype=tf.float32)
            background_class_tensor = tf.cast(background_class_tensor, dtype=tf.float32)

            # A weird "gotcha" is the negative/background class. Because our padding boxes are labeled as background,
            # these will slip in on the reduce_any.
            # Shape = LH X LW X #Anchor boxes X #Labels
            class_tensor = tf.concat([background_class_tensor, positive_class_tensor], axis=3)

            # Need more transposition magic to make broadcasting happy
            # Shape = [4 X ] LH X LW X #Anchor boxes X #Ground truth boxes
            layer_offset_tensor = tf.transpose(layer_offset_tensor, [4, 0, 1, 2, 3])
            match_tensor = tf.transpose(match_tensor, [1, 2, 0, 3])

            # Transpose back to normal
            # Shape = LH X LW X #Anchor boxes X #Ground truth boxes X 4
            layer_offset_tensor = tf.transpose(layer_offset_tensor * match_tensor, [1, 2, 3, 4, 0])

            # To combine the box offsets, we use the largest offsets in a given cell. Note: multiple objects in a given
            # cell should be rare, though could also cause strange behavior (such as only getting a single bounding box)
            # Shape = LH X LW X #Anchor boxes X 4
            layer_offset_tensor = tf.reduce_max(layer_offset_tensor, axis=3)

            # Concatenate, and we are done with this layer
            layer_target = tf.concat([layer_offset_tensor, class_tensor], axis=3)
            pred_layer_targets.append(layer_target)

        return tf.cast(image, tf.float32), tuple(pred_layer_targets)


def cast_image_to_float(image, bbox=[]):
    return tf.cast(image, tf.float32)


def calc_iou(ground_truth_boxes, default_box):
    """
    Helper function to compute IOU of a ground truth box and a given anchor/default box. Here many ground truth boxes
    are passed in at a time, though only a single default box. This is mainly due to the needs of the SSD encoding
    function.

    Note: The order of box parameters is different here than above because I got a choice in this case.
    xmin, xmax, ymin, ymax makes intuitive sense, the order from the TFRecord does not. In the future the parser ought
    to change so that this nice order is used throughout, but that is a task for another day. Apologies if the change
    in convention here adds any confusion.

    :param ground_truth_boxes: a list of tensors of the ground truth boxes for a given input image. The list elements
                               correspond to (xmin, xmax, ymin, ymax), and each of the 4 tensors has the following
                               Shape = Layer Height X Layer Width X Number of Bounding Boxes
    :param default_box: a list of 2D tensors describing the current anchor/default box at each spatial position
                        Shape = Layer Height X Layer Width
    :return: a tensor containing the IOU for each ground truth box should the given anchor box be applied to each
             spatial position.
    """
    # Extract the meaningful tensors from the arguments
    gt_xmin = tf.transpose(ground_truth_boxes[0], [2, 0, 1])
    gt_xmax = tf.transpose(ground_truth_boxes[1], [2, 0, 1])
    gt_ymin = tf.transpose(ground_truth_boxes[2], [2, 0, 1])
    gt_ymax = tf.transpose(ground_truth_boxes[3], [2, 0, 1])
    db_xmin = default_box[0]
    db_xmax = default_box[1]
    db_ymin = default_box[2]
    db_ymax = default_box[3]

    # Determine the (x, y)-coordinates of the intersection rectangle
    inter_xmin = tf.maximum(gt_xmin, db_xmin)
    inter_ymin = tf.maximum(gt_ymin, db_ymin)
    inter_xmax = tf.minimum(gt_xmax, db_xmax)
    inter_ymax = tf.minimum(gt_ymax, db_ymax)

    # compute the area of intersection rectangle
    inter_area = tf.maximum(0., inter_xmax - inter_xmin + 1.) * tf.maximum(0., inter_ymax - inter_ymin + 1.)

    # compute the area of both the prediction and ground-truth rectangles
    gt_area = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.)
    db_area = (db_xmax - db_xmin + 1.) * (db_ymax - db_ymin + 1.)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / (gt_area + db_area - inter_area)

    # return the intersection over union value
    return tf.transpose(iou, [1, 2, 0])
