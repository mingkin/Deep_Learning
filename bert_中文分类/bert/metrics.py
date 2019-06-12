# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : metrics.py
# Time    : 2019/6/12 0012 下午 2:14
© 2019 Ming. All rights reserved. Powered by King
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def cal_precision_recall(matrix, n_class, o_class):
    tn = matrix[n_class, n_class]
    fn = 0
    for i in o_class:
        fn += matrix[i, n_class]
    total = sum(matrix[n_class])
    recall = 0 if total == 0 else tn / (total + 0.0)
    precision = 0 if (tn + fn == 0) else tn / (tn + fn + 0.0)
    fp = (total - tn)

    return tn, fn, fp, precision, recall


def get_multi_metrics_num(matrix):
    # 0 正常
    tn0, fn0, fp0, precision0, recall0 = cal_precision_recall(matrix, 0, [1, 2])
    # 1 广告
    tn1, fn1, fp1, precision1, recall1 = cal_precision_recall(matrix, 1, [0, 2])
    # 2 色情
    tn2, fn2, fp2, precision2, recall2 = cal_precision_recall(matrix, 2, [0, 1])

    total = 0
    accuracy = 0
    for i in range(len(matrix)):
        accuracy += matrix[i, i]
        total += sum(matrix[i])

    accuracy = 0 if total == 0 else accuracy / float(total)

    fp = matrix[0, 1] + matrix[0, 2]
    fn = matrix[1, 0] + matrix[2, 0]
    tp = matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2]
    tn = total - tp - fp - fn
    precision = 0 if (tp + fp) == 0 else float(tp) / (tp + fp)
    recall = 0 if (tp + fn) == 0 else float(tp) / (tp + fn)

    metric = (
        accuracy, tp, fn, tn, fp, precision, recall,
        tn0, fp0, fn0, precision0, recall0, tn1, fp1, fn1, precision1, recall1, tn2, fp2, fn2, precision2, recall2
    )
    return metric


def get_multi_metrics(matrix):
    return format_multi_metrics(get_multi_metrics_num(matrix))


def format_multi_metrics(metric):
    return '\nAccuracy=%s,tp=%s,fn=%s,tn=%s,fp=%s,precision=%s,recall=%s\nLabel-0:tn=%s,fp=%s,fn=%s,Precision=%s,Recall=%s;\nLabel-1:tn=%s,fp=%s,fn=%s,Precision=%s,Recall=%s;\nLabel-2:tn=%s,fp=%s,fn=%s,Precision=%s,Recall=%s;' % metric


def get_metrics_ops(labels, predictions, num_labels):
    # 得到混淆矩阵和update_op，在这里我们需要将生成的混淆矩阵转换成tensor
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))

    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    # 得到numpy类型的混淆矩阵，然后计算precision，recall，f1值。
    precisions = []
    recalls = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1




