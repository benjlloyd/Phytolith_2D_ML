import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names,
                          title='Confusion matrix',
                          cmap=plt.cm.jet, axis_label=True, color_bar=True, ticks=True):
    counts = cm.sum(axis=1)[:, np.newaxis]
    cm = np.divide(cm.astype(np.float32), counts, out=np.zeros_like(cm), where=counts != 0)

    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if color_bar:
        fig.colorbar(im)
    ax.grid(True)

    if ticks:
        tick_marks = np.arange(len(class_names))
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    ax.set_title(title)

    if axis_label:
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    return fig, ax

