import sys
import gflags
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from .train_fixture import train_simple
from . import args
from .args import g
from ..common.utils import make_nets_and_opts
from .datasets import Dataset


if __name__ == '__main__':
    torch.manual_seed(931238)
    torch.set_num_threads(1)

    gflags.DEFINE_string('loss', 'cross_entropy', 'Can be cross_entropy, l1 or l2')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    # Populate args with additional commandline arguments
    args.fill(FLAGS.FlagDict())

    g.model_spec = yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader)

    print('global args:')
    print(args.repr())

    vis = SummaryWriter(comment=g.vis_tag)

    nets = make_nets_and_opts(g.model_spec, g.train_device)

    if g.random_rotation:
        # None means uniform rotation in [0, 2pi]
        angles = True if g.angles == '' else [float(t) for t in g.angles.split(',')]
    else:
        angles = False

    options = {}
    if g.random_rotation:
        options['random_rotation'] = angles
    if g.random_scale:
        options['random_scale'] = (g.scale_min, g.scale_max)

    dataset = Dataset(g.image_dir, g.image_size, g.mask_dir, **options)

    train_simple(
        nets={
            name: spec['net'] for name, spec in nets.items()
        },
        net_opts={
            name: spec['opt'] for name, spec in nets.items() if 'opt' in spec
        },
        vis=vis,
        train_dataset=dataset,
        global_args=g
    )
