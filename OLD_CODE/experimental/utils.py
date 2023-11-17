from LoadData import SubfamilyLoader, TribeLoader, GenusLoader
import numpy as np


def save_classes(class_dict, out_file):
    l = class_dict.items()
    l.sort(key=lambda e: e[1])
    with open(out_file, 'w') as f:
        for text, _ in l:
            f.write(text + '\n')


def load_classes(filename):
    with open(filename) as f:
        return filter(lambda x: x != '', [l.strip() for l in f.readlines()])


def get_dataset_loader(task):
    if task == 'subfamily':
        return SubfamilyLoader
    elif task == 'tribe':
        return TribeLoader
    elif task == 'genus':
        return GenusLoader
    else:
        raise RuntimeError('Cannot find loader for task %r' % task)


def _load_dataset(task, style, img_size, align_orientation=False):
    transforms = None
    if align_orientation:
        transforms = {}
        with open('../../data/align_transforms.txt') as f:
            for l in f:
                tokens = [e.strip() for e in l.split(',')]
                transform = np.array([float(e) for e in tokens[1:]]).reshape((3, 3))
                transforms[tokens[0]] = transform

    loader = get_dataset_loader(task)(img_size)

    img_dir = '../../data/%s' % style
    train_list_file = '../../data/%s_train.txt' % task
    test_list_file = '../../data/%s_test.txt' % task

    train_images, train_labels, class_dict = loader.load(img_dir, train_list_file, None, transforms)
    test_images, test_labels, _ = loader.load(img_dir, test_list_file, class_dict, transforms)

    return train_images, test_images, train_labels, test_labels, class_dict


def load_dataset(task, styles, img_size, align_orientation=False):
    all_train_images = []
    all_test_images = []
    all_train_labels = []
    all_test_labels = []

    for style in styles:
        train_images, test_images, train_labels, test_labels, class_dict = _load_dataset(
            task, style, img_size, align_orientation)

        all_train_images.append(train_images)
        all_test_images.append(test_images)
        all_train_labels.append(train_labels)
        all_test_labels.append(test_labels)

    all_train_images = np.array(all_train_images)
    all_test_images = np.array(all_test_images)
    all_train_labels = np.array(all_train_labels)
    all_test_labels = np.array(all_test_labels)

    return all_train_images, all_train_labels, all_test_images, all_test_labels
