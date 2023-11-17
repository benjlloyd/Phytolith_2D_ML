import numpy as np
import matplotlib.pyplot as plt
from . import plotting


def confusion_matrix(gt, preds, n_class):
    '''
    :return: n_class x n_class matrix. entry (i, j) is the number of predictions with label i given ground truth label j
    '''
    assert len(gt) == len(preds)
    m = np.zeros((n_class, n_class))

    for i in range(len(gt)):
        m[preds[i], gt[i]] += 1

    return m


def per_class_precision_recall(gt, preds, n_class):
    gt_items_per_class = [0] * n_class
    true_pos_per_class = [0] * n_class
    false_pos_per_class = [0] * n_class

    assert len(gt) == len(preds)

    for i in range(len(gt)):
        gt_items_per_class[gt[i]] += 1
        if preds[i] == gt[i]:
            true_pos_per_class[gt[i]] += 1
        else:
            false_pos_per_class[preds[i]] += 1

    res = []
    for i in range(n_class):
        if true_pos_per_class[i] + false_pos_per_class[i] == 0 or gt_items_per_class[i] == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = float(true_pos_per_class[i]) / (true_pos_per_class[i] + false_pos_per_class[i])
            recall = float(true_pos_per_class[i]) / gt_items_per_class[i]

        res.append((precision, recall, false_pos_per_class[i]))

    return res


def analyze_result(result, class_dict):
    """
    :param class_dict: string -> integer map
    :return:
    """
    labels = result['labels']
    preds = result['preds']

    class_array = [0] * len(class_dict)
    for k, v in class_dict.items():
        class_array[v] = k

    print('ground truth counts')
    for i in range(len(class_array)):
        print(class_array[i], np.sum((np.array(labels) == i).astype(np.int)))

    prec_recall = per_class_precision_recall(labels, preds, len(class_dict))
    print('class,precision,recall,false_pos')
    for i in range(len(class_array)):
        print('%s,%f,%f,%d' % (class_array[i], prec_recall[i][0], prec_recall[i][1], prec_recall[i][2]))

    confusion_mat = confusion_matrix(labels, preds, len(class_dict))

    plotting.plot_confusion_matrix(confusion_mat, class_array)
    plt.draw()
    plt.show()
