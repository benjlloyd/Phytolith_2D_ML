'''
Commonly used default arguments
'''


from easydict import EasyDict as edict
import gflags
import tabulate


# Training parameters
gflags.DEFINE_string('model_spec', '', 'Path to the model specification file.')
gflags.DEFINE_string('model_file', 'model.th', '')
gflags.DEFINE_string('image_dir', '', '')
gflags.DEFINE_integer('image_size', 125, '')
gflags.DEFINE_string('mask_dir', '', '')
gflags.DEFINE_boolean('random_rotation', False, '')
gflags.DEFINE_string('angles', '',
                     'additional flag for random_rotation. '
                     'a list of angles, otherwise assume uniform rotation in [0, 2*pi] (if random_rotation=True).')
gflags.DEFINE_boolean('random_scale', False, '')
gflags.DEFINE_float('scale_min', 0.5, '')
gflags.DEFINE_float('scale_max', 2.0, '')

gflags.DEFINE_integer('n_dataset_worker', 1, '')
gflags.DEFINE_integer('log_interval', 1000, 'Interval for logging.')
gflags.DEFINE_integer('vis_interval', 1, 'Interval for updating visualization.')
gflags.DEFINE_string('vis_tag', '', 'This is passed to SummaryWriter as the comment kwargs.')
gflags.DEFINE_integer('save_interval', 200, 'Epoch interval for saving model weights to file.')
gflags.DEFINE_string('train_device', 'cuda', 'Training device. See PyTorch doc for the format.')


# Model hyper parameters
gflags.DEFINE_integer('batch_size', 32, '')
gflags.DEFINE_integer('max_epochs', 30, '')
gflags.DEFINE_integer('lr_decay_epoch', 100, '')
gflags.DEFINE_float('lr_decay_rate', 0.7, '')


defaults = {}


def helper(d):
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret[k] = helper(v)
        else:
            ret[k] = v
    return tabulate.tabulate(ret.items())


# The global flag tree. Tree can be augmented by loading external config files.
g = edict(defaults)


# Since g's values can be accessed as attributes, we cannot add additional methods to it.
# Use the following methods to manipulate g.


def fill(args):
    for key in args.keys():
        g[key] = args[key].value


def set_s(set_str):
    ss = set_str.split(',')
    for s in ss:
        if s == '':
            continue
        field, value = s.split('=')
        try:
            value = eval(value, {'__builtins__': None})
        except:
            # Cannot convert the value. Treat it as it is.
            pass
        attrs = field.split('.')
        node = g
        for i in range(len(attrs) -1):
            node = node[attrs[i]]
        node[attrs[-1]] = value


def repr(fmt='plain'):
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items(), tablefmt=fmt)
    return helper(g)
