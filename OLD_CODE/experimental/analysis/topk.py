from utils import load_classes
import numpy as np


TASK = 'genus'
MODEL = 'vgg19'
OPTION = 'unweighted'
STYLE = 'mean_global_autocontrast_normalized'


class TopK:
    def __init__(self, task, model, option, style):

        self.CONFUSION_MATRIX_FILE = '../../data/%s_classification_results/%s/%s/confusion_matrix/%s_confusion_mat.txt' % (
            task, model, option, style
        )

        self.SCORE_FILE = '../../data/%s_classification_results/%s/%s/scores/%s_scores.txt' % (
            task, model, option, style
        )

        self.TEST_LABEL_FILE = '../../data/%s_test_labels.txt' % (
            task
        )

        self.confmat = np.genfromtxt(self.CONFUSION_MATRIX_FILE, dtype=np.int32, delimiter=',')
        self.scores = np.genfromtxt(self.SCORE_FILE, dtype=np.float32, delimiter=',')
        self.classes = load_classes('../../data/%s_names.txt' % TASK)
        self.labels = np.genfromtxt(self.TEST_LABEL_FILE, dtype=np.int32)

    def topk(self, k):
        n_correct = 0
        for i in xrange(self.scores.shape[0]):
            preds = np.argsort(self.scores[i])[::-1]
            for j in xrange(k):
                if preds[j] == self.labels[i]:
                    n_correct += 1
        return n_correct / float(len(self.labels))


if __name__ == '__main__':
    styles = [
        'max_global_autocontrast_normalized',
        'mean_global_autocontrast_normalized',
        'global_autocontrast_normalized',
        'focus_stacking_global_autocontrast_normalized',

        'max_images',
        'mean_images',
        'median_images',
        'focus_stacking_images'
    ]
    for style in styles:
        genus_topk = TopK('genus', 'vgg19', 'unweighted', style)
        print style
        print 'top1:', genus_topk.topk(1)
        print 'top3:', genus_topk.topk(3)
        print 'top5:', genus_topk.topk(5)
