"""补充内容见evaluate.ipynb"""
from __future__ import division

import numpy as np
import six


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    pred_labels = iter(pred_labels)     # (352, 480)
    gt_labels = iter(gt_labels)     # (352, 480)

    n_class = 3
    confusion = np.zeros((n_class, n_class), dtype=np.int64)    # (12, 12)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()   # (168960, )
        gt_label = gt_label.flatten()   # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.  极度巧妙 × class_nums 正好使得每个ij能够对应.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2)\
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]
    # return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`

    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    # Evaluation code is based on
    # https://github.com/shelhamer/model.berkeleyvision.org/blob/master/
    # score.py#L37
    """

    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)     # (11, )
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)
    precision = np.diag(confusion) / confusion.sum(axis = 1)
    recall = np.diag(confusion) / confusion.sum(axis = 0)
    
    f1score = 2 * precision * recall / (precision + recall)
    intersection = np.diag(confusion)  
    union = np.sum(confusion, axis = 1) + np.sum(confusion, axis = 0) - np.diag(confusion)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    intersection = 2*np.diag(confusion)  
    union = np.sum(confusion, axis = 1) + np.sum(confusion, axis = 0) 
    dice = intersection / union
    dice = np.nanmean(dice) 
    

    return {'iou': iou, 'miou': mIoU,
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1]),
            'f1score': np.nanmean(f1score),
            'recall':np.nanmean(recall),
            'DICE':dice,
            }
            # 'mean_class_accuracy': np.nanmean(class_accuracy)}