import torch
import numpy as np
import yaml
from .common.utils import make_nets, pprint_dict


def _load(weights_file, device='cuda'):
    state_dict = torch.load(weights_file, map_location='cpu')
    print('loaded %s' % weights_file)
    g = state_dict.get('global_args', {})
    print('global args:')
    print(pprint_dict(g))

    if isinstance(g.model_spec, dict):
        nets = make_nets(g.model_spec, device)
    else:
        nets = make_nets(yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader), device)

    for name, net in nets.items():
        net.load_state_dict(state_dict['nets'][name])
        net.train(False)

    return nets, g, state_dict['class_dict']


class ClassifierBase(object):
    def __init__(self, weights_file, device='cuda'):
        self.nets, self.g, self.class_dict = _load(weights_file, device)

        self.inverse_class_dict = dict()
        for name, label in self.class_dict.items():
            self.inverse_class_dict[label] = name

        self.device = device

    def _predict(self, nets, device, img):
        raise NotImplementedError

    def predict(self, img):
        return self._predict(self.nets, self.device, img)


class _SimpleForward(object):
    def _predict(self, nets, device, img):
        # Input: 1 x H x W or 3 x H x W
        with torch.no_grad():
            img_th = torch.as_tensor(img, dtype=torch.float, device=device).unsqueeze(0)
            logits = nets['classifier'](nets['feature_extractor'](img_th))
            probs = torch.softmax(logits, dim=1)
            return probs[0].data.cpu().numpy()


class ClassifierSimple(_SimpleForward, ClassifierBase):
    pass


class ClassifierEnsembleOutputAverageBase(object):
    def __init__(self, weights_file_list, device='cuda'):
        self.classifiers = [_load(_, device) for _ in weights_file_list]
        self.device = device

        _, self.g, self.class_dict = self.classifiers[0]

        self.inverse_class_dict = dict()
        for name, label in self.class_dict.items():
            self.inverse_class_dict[label] = name

    def _predict(self, nets, device, img):
        raise NotImplementedError

    def predict(self, img):
        probs = [self._predict(nets, self.device, img) for nets, _, _ in self.classifiers]
        return np.mean(probs, axis=0)

    def predict_raw(self, img):
        return [self._predict(nets, self.device, img) for nets, _, _ in self.classifiers]


class ClassifierEnsembleOutputAverageSimple(_SimpleForward, ClassifierEnsembleOutputAverageBase):
    pass


if __name__ == '__main__':
    import cv2
    import numpy as np
    from .preprocess.normalization import resize_and_crop

    def test_classifier_simple():
        classifier = ClassifierSimple('experiments/resnet_finetune/species/unfreeze_last_4/randrot-randshift0.2-breflect-model.300')
        img = cv2.imread('../../data/new_dataset_2019/normalized/puelioideae.guaduellieae.guaduella.macrostachys.8.tif',
                         cv2.IMREAD_GRAYSCALE)
        img = resize_and_crop(img, 224, border_mode='reflect')
        img = (img / 255.0).astype(np.float32)
        probs = classifier.predict(img[None, :, :])

        top_indices = np.argsort(probs)[::-1]

        for i in range(5):
            print(classifier.inverse_class_dict[top_indices[i]], probs[top_indices[i]])

    def test_classifier_ensemble():
        weights_files = [
            'experiments/resnet_finetune/tribe/unfreeze_last_4/randrot-randshift0.2-breflect-weighted-fold%d-model.300'
            % i for i in range(9)
        ]

        classifier = ClassifierEnsembleOutputAverageSimple(weights_files)
        img = cv2.imread('/mnt/ssd1/phytoliths/new_dataset_2019_normalized/fossils/fossils_1/ANK04-103 (1).TIF',
                         cv2.IMREAD_GRAYSCALE)
        img = resize_and_crop(img, 224, border_mode='reflect')
        img = (img / 255.0).astype(np.float32)
        probs = classifier.predict(img[None, :, :])

        top_indices = np.argsort(probs)[::-1]

        for i in range(5):
            print(classifier.inverse_class_dict[top_indices[i]], probs[top_indices[i]])

    test_classifier_ensemble()
