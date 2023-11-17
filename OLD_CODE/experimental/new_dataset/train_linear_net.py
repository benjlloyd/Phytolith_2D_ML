import sys
import gflags
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from .train_fixture import train_simple, test_simple, train_metric
from .analysis.utils import analyze_result
from . import args
from .args import g
from .common.utils import make_nets_and_opts, create_datasets


if __name__ == '__main__':
    gflags.DEFINE_string('model_variant', 'default', '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    # Populate args with additional commandline arguments
    args.fill(FLAGS.FlagDict())

    g.model_spec = yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader)

    print('global args:')
    print(args.repr())

    torch.manual_seed(g.torch_seed)
    torch.set_num_threads(1)

    vis = SummaryWriter(comment=g.vis_tag)
    vis.add_text('global args', args.repr())

    nets = make_nets_and_opts(g.model_spec, g.train_device)
    train_dataset, validation_dataset, test_dataset = create_datasets(g)

    if g.metric:
        train_func = train_metric
    else:
        train_func = train_simple

    train_func(
        nets={
            name: spec['net'] for name, spec in nets.items()
        },
        net_opts={
            name: spec['opt'] for name, spec in nets.items() if 'opt' in spec
        },
        vis=vis,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        global_args=g
    )

    # Analysis on test result
    # print('testing...')
    # test_result = test_simple(nets={
    #         name: spec['net'] for name, spec in nets.items()
    #     },
    #     dataset=test_dataset, vis=None, global_step=None, global_args=g)
    #
    # analyze_result(test_result, test_dataset.class_dict)
