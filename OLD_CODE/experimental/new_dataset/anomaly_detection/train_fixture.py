import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import tabulate
from ..common.utils import module_grad_stats, save_model


def train_simple(nets, net_opts, vis, train_dataset, global_args):
    """
    :param nets: a dict containing network parts.
    :param net_opts: a dict containing optimizers for each network part.
    :param vis: SummaryWriter object for visualization.
    :return:
    """
    (
        model_file,
        max_epochs,
        batch_size,
        n_worker,
        log_interval,
        vis_interval,
        save_interval,
        train_device,
        loss_type,
    ) = [global_args[_] for _ in ['model_file',
                                  'max_epochs',
                                  'batch_size',
                                  'n_dataset_worker',
                                  'log_interval',
                                  'vis_interval',
                                  'save_interval',
                                  'train_device',
                                  'loss'
                                  ]]
    epoch = 0
    last_epoch = -1

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=global_args['lr_decay_epoch'],
            gamma=global_args['lr_decay_rate'],
            last_epoch=last_epoch)
        for name, opt in net_opts.items()
    }

    while True:
        print('===== epoch %d =====' % epoch)

        sampler = RandomSampler(train_dataset)
        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=n_worker,
                            pin_memory=True,
                            drop_last=True)

        for idx, (batch_img, batch_mask) in enumerate(loader):
            for _, opt in net_opts.items():
                opt.zero_grad()

            batch_img = batch_img.to(device=train_device, non_blocking=True)
            batch_mask = batch_mask.to(device=train_device, non_blocking=True)
            batch_size = batch_img.size(0)

            features = nets['encoder'](batch_img)
            reconstruction_logits = nets['decoder'](features)
            reconstruction = torch.sigmoid(reconstruction_logits)

            if loss_type == 'cross_entropy':
                # batch_size x 1 x H x W
                loss = F.binary_cross_entropy_with_logits(reconstruction_logits, batch_img, reduction='none')
                loss = torch.mean(torch.sum((loss * batch_mask).view(batch_size, -1), dim=1))
            elif loss_type == 'l1':
                loss = torch.mean(torch.sum(torch.abs(reconstruction * batch_mask - batch_img).view(batch_size, -1), dim=1))
            elif loss_type == 'l2':
                loss = torch.mean(torch.sum(torch.pow(reconstruction * batch_mask - batch_img, 2).view(batch_size, -1), dim=1))
            else:
                raise RuntimeError('Unsupported loss type %s' % loss_type)

            loss.backward()

            for _, opt in net_opts.items():
                opt.step()

            if idx % log_interval == 0:
                print('step %d loss %.3f' % (idx, loss.item()))
                lrs = [(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]
                print('learning rate:\n%s' % tabulate.tabulate(lrs))
                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, module_grad_stats(net)))

                global_step = idx * batch_size + epoch * len(train_dataset)
                vis.add_scalar('loss/train', loss.item(), global_step)
                vis.add_scalars('learning_rate', dict(lrs), global_step)

                vis.add_images('input_img', batch_img, global_step)

                vis.add_images('reconstruction', reconstruction.data, global_step)
                vis.add_images('mask', batch_mask.data, global_step)

                l2 = torch.mean(torch.sum(torch.pow(reconstruction.data - batch_img, 2).view(batch_size, -1), dim=1))
                vis.add_scalar('l2_diff', l2.item(), global_step)

                l1 = torch.mean(torch.sum(torch.abs(reconstruction.data - batch_img).view(batch_size, -1), dim=1))
                vis.add_scalar('l1_diff', l1.item(), global_step)

        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1

        if epoch % save_interval == 0 or epoch == max_epochs:
            print('saving model...')
            state = {
                'epoch': epoch,
                'global_args': global_args,
                'optims': {
                    name: opt.state_dict() for name, opt in net_opts.items()
                },
                'nets': {
                    name: net.state_dict() for name, net in nets.items()
                }
            }
            save_model(state, epoch, '', model_file)

        if epoch >= max_epochs:
            break
