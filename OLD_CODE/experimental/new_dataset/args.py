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
# gflags.DEFINE_string('mask_dir', '', '')
gflags.DEFINE_integer('image_size', 224, '')
gflags.DEFINE_string('train_list_file', '', '')
gflags.DEFINE_string('validation_list_file', '', '')
gflags.DEFINE_string('test_list_file', '', '')
gflags.DEFINE_string('granularity', 'subfamily', '')
gflags.DEFINE_string('border_mode', 'reflect', 'Can be none, zero, replicate or reflect.')
gflags.DEFINE_boolean('random_rotation', False, '')
gflags.DEFINE_string('angles', '',
                     'additional flag for random_rotation. '
                     'a list of angles, otherwise assume uniform rotation in [0, 2*pi] (if random_rotation=True).')
gflags.DEFINE_boolean('random_scale', False, '')
gflags.DEFINE_float('scale_min', 0.5, '')
gflags.DEFINE_float('scale_max', 2.0, '')
gflags.DEFINE_boolean('random_shift', False, '')
gflags.DEFINE_float('shift_min', -0.1, 'Min shift ratio to image size.')
gflags.DEFINE_float('shift_max', 0.1, 'Max shift ratio to image size.')
gflags.DEFINE_boolean('metric', False, 'Train with metric learning.')
gflags.DEFINE_float('metric_margin', 10.0, '')
gflags.DEFINE_float('metric_loss_coeff', 1.0, 'Coefficient for metric loss.')
gflags.DEFINE_string('metric_training_method', 'together', 'Can be together, altstep, altepoch')
gflags.DEFINE_boolean('weighted_sampling', False, 'Sample each class with equal probability.')
gflags.DEFINE_boolean('weighted_loss', False, '')
gflags.DEFINE_boolean('augment_val_set', False, 'Whether apply data augmentation to validation set.')


gflags.DEFINE_integer('torch_seed', 931238, 'pytorch manual seed.')
gflags.DEFINE_integer('n_dataset_worker', 4, '')
gflags.DEFINE_integer('log_interval', 1000, 'Interval for logging.')
gflags.DEFINE_integer('vis_interval', 1000, 'Interval for updating visualization.')
gflags.DEFINE_string('vis_tag', '', 'This is passed to SummaryWriter as the comment kwargs.')
gflags.DEFINE_integer('save_interval', 250, 'Epoch interval for saving writing model weights to file.')
gflags.DEFINE_string('train_device', 'cuda', 'Training device. See PyTorch doc for the format.')


# Model hyper parameters
gflags.DEFINE_integer('batch_size', 32, '')
gflags.DEFINE_integer('max_epochs', 1000, '')
gflags.DEFINE_integer('lr_decay_epoch', 50, '')
gflags.DEFINE_float('lr_decay_rate', 0.7, '')


# Metric Learning
# TODO: move this to metric learning specific files
# gflags.DEFINE_integer('metric_max_epochs', 30, '')
# gflags.DEFINE_float('metric_learning_rate', 1e-5, '')
# gflags.DEFINE_integer('metric_lr_decay_steps', 20, '')
# gflags.DEFINE_float('metric_lr_decay_rate', 0.3, '')


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
