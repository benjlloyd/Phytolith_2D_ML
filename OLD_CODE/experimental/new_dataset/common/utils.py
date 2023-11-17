from . import datasets, models
import os
from torch import optim
import tabulate


def module_grad_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param.grad), maybe_min(param.grad))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def make_nets_and_opts(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(models, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        d = {
            'net': net,
        }
        if 'opt' in spec:
            d['opt'] = getattr(optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])

        ret[net_name] = d
    return ret


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(models, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        ret[net_name] = net
    return ret


def create_datasets(g):
    if g.random_rotation:
        angles = True if g.angles == '' else [float(t) for t in g.angles.split(',')]
    else:
        angles = False

    options = {}
    if g.random_rotation:
        options['random_rotation'] = angles
    if g.random_scale:
        options['random_scale'] = (g.scale_min, g.scale_max)
    if g.random_shift:
        options['random_shift'] = (g.shift_min, g.shift_max)

    train_dataset = datasets.Dataset(
        g.image_dir,
        g.train_list_file,
        g.granularity,
        g.image_size,
        border_mode=g.border_mode,
        pair=g.metric,
        **options)

    class_dict = train_dataset.class_dict

    if g.augment_val_set:
        val_options = options
    else:
        val_options = dict()

    validation_dataset = datasets.Dataset(
        g.image_dir,
        g.validation_list_file,
        g.granularity,
        g.image_size,
        class_dict,
        border_mode=g.border_mode,
        **val_options)

    test_dataset = datasets.Dataset(
        g.image_dir,
        g.test_list_file,
        g.granularity,
        g.image_size,
        class_dict,
        border_mode=g.border_mode)

    print('%d training images' % len(train_dataset))
    print('%d validation images' % len(validation_dataset))
    print('%d test images' % len(test_dataset))
    print('%d classes' % len(class_dict))

    return train_dataset, validation_dataset, test_dataset


def save_model(state, step, dir, filename):
    import torch
    path = os.path.join(dir, '%s.%d' % (filename, step))
    torch.save(state, path)


def pprint_dict(x):
    """
    :param x: a dict
    :return: a string of pretty representation of the dict
    """
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items())
    return helper(x)


class TaxonomicRank(object):
    def __init__(self, train_list_file):
        """
        :param train_list_file: a text file consisting of training images. Taxonomic information is extracted
                                from the file names.
        """
        subfamilies = dict()
        tribes = dict()
        genus = dict()
        species = dict()

        with open(train_list_file) as f:
            for l in f:
                tokens = l.split('.')
                _1, _2, _3, _4 = tokens[:4]
                subfamilies[_1] = (_1, _2, _3, _4)
                tribes[_2] = (_1, _2, _3, _4)
                genus[_3] = (_1, _2, _3, _4)
                species[_4] = (_1, _2, _3, _4)

        self.subfamily = subfamilies
        self.tribe = tribes
        self.genus = genus
        self.species = species
