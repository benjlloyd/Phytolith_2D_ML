import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import tabulate
from .common.utils import module_grad_stats, save_model


def train_simple(nets, net_opts, vis, train_dataset, validation_dataset, global_args):
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
        weighted_sampling,
        weighted_loss,
        model_variant
    ) = [global_args[_] for _ in ['model_file',
                                  'max_epochs',
                                  'batch_size',
                                  'n_dataset_worker',
                                  'log_interval',
                                  'vis_interval',
                                  'save_interval',
                                  'train_device',
                                  'weighted_sampling',
                                  'weighted_loss',
                                  'model_variant'
                                  ]]

    assert not (weighted_sampling and weighted_loss)  # It doesn't make sense to enable both

    if weighted_loss:
        class_weight = torch.as_tensor(train_dataset.normalized_inv_label_freq, device=train_device)
    else:
        class_weight = None

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

        if weighted_sampling:
            sampler = WeightedRandomSampler(train_dataset.label_weights, len(train_dataset), replacement=True)
        else:
            sampler = RandomSampler(train_dataset)

        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=n_worker,
                            pin_memory=True,
                            drop_last=True)

        n_correct = 0
        n_total = 0
        for idx, (batch_img, batch_label) in enumerate(loader):
            for _, opt in net_opts.items():
                opt.zero_grad()

            batch_img = batch_img.to(device=train_device, non_blocking=True)
            batch_label = batch_label.to(device=train_device, non_blocking=True)

            if model_variant == 'default':
                features = nets['feature_extractor'](batch_img)
                logits = nets['classifier'](features)
                loss = F.cross_entropy(logits, batch_label, weight=class_weight)

            elif model_variant == 'attention':
                attention_feature = nets['attention_encoder'](batch_img)
                attention_weights = torch.sigmoid(nets['attention_decoder'](attention_feature) - 1.0)
                assert attention_weights.size() == batch_img.size()
                features = nets['feature_extractor'](batch_img * attention_weights)
                logits = nets['classifier'](features)
                loss = F.cross_entropy(logits, batch_label, weight=class_weight)

            else:
                raise RuntimeError('Unknown model variant %s' % model_variant)

            loss.backward()
            for _, opt in net_opts.items():
                opt.step()

            _, pred_label = torch.max(logits, 1)
            n_correct += torch.sum(pred_label == batch_label).item()
            n_total += batch_label.size(0)

            global_step = idx * batch_size + epoch * len(train_dataset)

            if idx % log_interval == 0:
                print('step %d loss %.3f' % (idx, loss.item()))
                lrs = [(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]
                print('learning rate:\n%s' % tabulate.tabulate(lrs))
                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, module_grad_stats(net)))

                vis.add_scalar('loss/train', loss.item(), global_step)
                vis.add_scalars('learning_rate', dict(lrs), global_step)

            if idx % vis_interval == 0:
                vis.add_images('batch_img', batch_img, global_step)

                if model_variant == 'attention':
                    vis.add_images('attention_weights', attention_weights, global_step)

        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1

        train_accuracy = n_correct / float(n_total)
        print('training accuracy: %.3f' % train_accuracy)
        vis.add_scalar('accuracy/train', train_accuracy, epoch * len(train_dataset))

        if validation_dataset is not None:
            validation_result = test_simple(nets, validation_dataset, vis, epoch * len(train_dataset), global_args)
            print('validation_accuracy: top1 %.3f top5 %.3f' %
                  (validation_result['accuracy'], validation_result['top5_accuracy']))
            vis.add_scalar('accuracy/validation', validation_result['accuracy'], epoch * len(train_dataset))
            vis.add_scalar('accuracy/validation-top5', validation_result['top5_accuracy'], epoch * len(train_dataset))

        if epoch % save_interval == 0 or epoch == max_epochs:
            print('saving model...')
            state = {
                'epoch': epoch,
                'global_args': global_args,
                'class_dict': train_dataset.class_dict,
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


def test_simple(nets, dataset, vis, global_step, global_args):
    (
        train_device,
        model_variant
    ) = [global_args[_] for _ in ['train_device', 'model_variant']]

    loader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True,
                        drop_last=False)

    n_correct = 0
    n_correct_top5 = 0
    n_total = 0

    all_labels = []
    all_preds = []

    for _, net in nets.items():
        net.train(False)

    for idx, (batch_img, batch_label) in enumerate(loader):
        with torch.no_grad():
            batch_img = batch_img.to(device=train_device, non_blocking=True)
            batch_label = batch_label.to(device=train_device, non_blocking=True)

            if model_variant == 'default':
                features = nets['feature_extractor'](batch_img)
                logits = nets['classifier'](features)

            elif model_variant == 'attention':
                attention_feature = nets['attention_encoder'](batch_img)
                attention_weights = torch.sigmoid(nets['attention_decoder'](attention_feature) - 1.0)
                assert attention_weights.size() == batch_img.size()
                features = nets['feature_extractor'](batch_img * attention_weights)
                logits = nets['classifier'](features)

            else:
                raise RuntimeError('Unsupported model variant %s' % model_variant)

            _, pred_top5_labels = torch.topk(logits, 5, dim=1)

            n_correct += torch.sum(pred_top5_labels[:, 0] == batch_label).item()
            n_correct_top5 += torch.sum(pred_top5_labels == batch_label.unsqueeze(1).expand_as(pred_top5_labels)).item()

            n_total += batch_label.size(0)
            all_labels.append(batch_label)
            all_preds.append(pred_top5_labels[:, 0])

            if idx == 0 and vis is not None:
                vis.add_images('test_batch', batch_img, global_step)

    for _, net in nets.items():
        net.train(True)

    return {
        'accuracy': n_correct / float(n_total),
        'top5_accuracy': n_correct_top5 / float(n_total),
        'labels': torch.cat(all_labels).data.cpu().numpy(),
        'preds': torch.cat(all_preds).data.cpu().numpy()
    }


def train_metric(nets, net_opts, vis, train_dataset, validation_dataset, global_args):
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
        weighted_sampling,
        metric_margin,
        metric_loss_coeff,
        training_method
    ) = [global_args[_] for _ in ['model_file',
                                  'max_epochs',
                                  'batch_size',
                                  'n_dataset_worker',
                                  'log_interval',
                                  'vis_interval',
                                  'save_interval',
                                  'train_device',
                                  'weighted_sampling',
                                  'metric_margin',
                                  'metric_loss_coeff',
                                  'metric_training_method'
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

    def make_sampler():
        if weighted_sampling:
            sampler = WeightedRandomSampler(train_dataset.label_weights, len(train_dataset), replacement=True)
        else:
            sampler = RandomSampler(train_dataset)

        return sampler

    while True:
        print('===== epoch %d =====' % epoch)

        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=make_sampler(),
                            num_workers=n_worker,
                            pin_memory=True,
                            drop_last=True)

        n_correct = 0
        n_total = 0

        if training_method == 'together':
            for idx, (batch_img1, batch_img2, batch_label1, batch_label2) in enumerate(loader):
                for _, opt in net_opts.items():
                    opt.zero_grad()

                batch_img1 = batch_img1.to(device=train_device, non_blocking=True)
                batch_img2 = batch_img2.to(device=train_device, non_blocking=True)
                batch_label1 = batch_label1.to(device=train_device, non_blocking=True)
                batch_label2 = batch_label2.to(device=train_device, non_blocking=True)

                features1 = nets['feature_extractor'](batch_img1)
                features2 = nets['feature_extractor'](batch_img2)

                logits1 = nets['classifier'](features1)
                logits2 = nets['classifier'](features2)

                # same label: 1, diff label: -1
                target = (batch_label1 == batch_label2).type(torch.float, non_blocking=True) * 2.0 - 1.0

                feature_distance = torch.norm(features1 - features2, p=2, dim=1)
                metric_loss = F.hinge_embedding_loss(feature_distance, target, margin=metric_margin)

                logits = torch.cat([logits1, logits2], dim=0)
                labels = torch.cat([batch_label1, batch_label2], dim=0).data
                classification_loss = F.cross_entropy(logits, labels)

                loss = metric_loss * metric_loss_coeff + classification_loss
                loss.backward()

                for _, opt in net_opts.items():
                    opt.step()

                _, pred_label = torch.max(logits.data, 1)
                n_correct += torch.sum(pred_label == labels).item()
                n_total += batch_label1.size(0) + batch_label2.size(0)

                if idx % log_interval == 0:
                    print('step %d classification_loss %.3f metric_loss %.3f' %
                          (idx, classification_loss.item(), metric_loss.item()))
                    lrs = [(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]
                    print('learning rate:\n%s' % tabulate.tabulate(lrs))
                    for name, net in nets.items():
                        print('%s grad:\n%s' % (name, module_grad_stats(net)))

                    global_step = idx * batch_size + epoch * len(train_dataset)
                    vis.add_scalar('loss/train', loss.item(), global_step)
                    vis.add_scalars('learning_rate', dict(lrs), global_step)
                    vis.add_images('batch_img1', batch_img1, global_step)

        elif training_method == 'altstep':
            for idx, (batch_img1, batch_img2, batch_label1, batch_label2) in enumerate(loader):
                net_opts['feature_extractor'].zero_grad()

                batch_img1 = batch_img1.to(device=train_device, non_blocking=True)
                batch_img2 = batch_img2.to(device=train_device, non_blocking=True)
                batch_label1 = batch_label1.to(device=train_device, non_blocking=True)
                batch_label2 = batch_label2.to(device=train_device, non_blocking=True)

                features1 = nets['feature_extractor'](batch_img1)
                features2 = nets['feature_extractor'](batch_img2)

                # same label: 1, diff label: -1
                target = (batch_label1 == batch_label2).type(torch.float, non_blocking=True) * 2.0 - 1.0

                feature_distance = torch.norm(features1 - features2, p=2, dim=1)
                metric_loss = F.hinge_embedding_loss(feature_distance, target.data, margin=metric_margin)
                metric_loss.backward()

                if idx % log_interval == 0:
                    print('%s grad:\n%s' % ('feature_extractor', module_grad_stats(nets['feature_extractor'])))

                net_opts['feature_extractor'].step()

                net_opts['classifier'].zero_grad()

                # Should we freeze the feature extractor?
                # By freezing the feature extractor, we effectively only train the classifier using metric-learned
                # features.
                with torch.no_grad():
                    features = nets['feature_extractor'](torch.cat([batch_img1, batch_img2], dim=0))

                logits = nets['classifier'](features)

                labels = torch.cat([batch_label1, batch_label2], dim=0)
                classification_loss = F.cross_entropy(logits, labels)
                classification_loss.backward()

                net_opts['classifier'].step()

                _, pred_label = torch.max(logits.data, 1)
                n_correct += torch.sum(pred_label == labels).item()
                n_total += batch_label1.size(0) + batch_label2.size(0)

                if idx % log_interval == 0:
                    print('%s grad:\n%s' % ('classifier', module_grad_stats(nets['classifier'])))
                    print('step %d classification_loss %.3f metric_loss %.3f' %
                          (idx, classification_loss.item(), metric_loss.item()))
                    lrs = [(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]
                    print('learning rate:\n%s' % tabulate.tabulate(lrs))

                    global_step = idx * batch_size + epoch * len(train_dataset)
                    vis.add_scalar('loss/train', metric_loss.item() + classification_loss.item(), global_step)
                    vis.add_scalars('learning_rate', dict(lrs), global_step)
                    vis.add_images('batch_img1', batch_img1, global_step)

        elif training_method == 'altepoch':
            for idx, (batch_img1, batch_img2, batch_label1, batch_label2) in enumerate(loader):
                net_opts['feature_extractor'].zero_grad()

                batch_img1 = batch_img1.to(device=train_device, non_blocking=True)
                batch_img2 = batch_img2.to(device=train_device, non_blocking=True)
                batch_label1 = batch_label1.to(device=train_device, non_blocking=True)
                batch_label2 = batch_label2.to(device=train_device, non_blocking=True)

                features1 = nets['feature_extractor'](batch_img1)
                features2 = nets['feature_extractor'](batch_img2)

                # same label: 1, diff label: -1
                # target = (batch_label1 == batch_label2).type(torch.float, non_blocking=True) * 2.0 - 1.0
                # feature_distance = torch.norm(features1 - features2, p=2, dim=1)
                # # feature_distance = torch.sum(torch.pow(features1 - features2, 2), dim=1)
                # assert feature_distance.size() == target.size()
                # metric_loss = F.hinge_embedding_loss(feature_distance, target, margin=metric_margin)

                # same label: 1, diff label: 0
                target = (batch_label1 == batch_label2).type(torch.float, non_blocking=True)
                feature_distance = torch.norm(features1 - features2, p=2, dim=1)
                assert feature_distance.size() == target.size()
                metric_loss = target * torch.pow(feature_distance, 2) + \
                              (1 - target) * torch.pow(torch.max(feature_distance.new_tensor([0.0]),
                                                                 metric_margin - feature_distance), 2)
                metric_loss = torch.mean(metric_loss)
                metric_loss.backward()

                net_opts['feature_extractor'].step()

                if idx % log_interval == 0:
                    print('step %d metric_loss %.3f' % (idx, metric_loss.item()))
                    global_step = idx * batch_size + epoch * len(train_dataset)
                    vis.add_scalar('loss/metric', metric_loss.item(), global_step)

            loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                sampler=make_sampler(),
                                num_workers=n_worker,
                                pin_memory=True,
                                drop_last=True)

            for idx, (batch_img1, batch_img2, batch_label1, batch_label2) in enumerate(loader):
                net_opts['classifier'].zero_grad()

                batch_img1 = batch_img1.to(device=train_device, non_blocking=True)
                batch_img2 = batch_img2.to(device=train_device, non_blocking=True)
                batch_label1 = batch_label1.to(device=train_device, non_blocking=True)
                batch_label2 = batch_label2.to(device=train_device, non_blocking=True)

                with torch.no_grad():
                    features = nets['feature_extractor'](torch.cat([batch_img1, batch_img2], dim=0))

                logits = nets['classifier'](features)

                labels = torch.cat([batch_label1, batch_label2], dim=0)
                classification_loss = F.cross_entropy(logits, labels)
                classification_loss.backward()

                net_opts['classifier'].step()

                _, pred_label = torch.max(logits.data, 1)
                n_correct += torch.sum(pred_label == labels).item()
                n_total += batch_label1.size(0) + batch_label2.size(0)

                if idx % log_interval == 0:
                    print('step %d classification_loss %.3f' %
                          (idx, classification_loss.item()))
                    lrs = [(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]
                    print('learning rate:\n%s' % tabulate.tabulate(lrs))
                    for name, net in nets.items():
                        print('%s grad:\n%s' % (name, module_grad_stats(net)))

                    global_step = idx * batch_size + epoch * len(train_dataset)
                    vis.add_scalar('loss/train', classification_loss.item(), global_step)
                    vis.add_scalars('learning_rate', dict(lrs), global_step)
                    vis.add_images('batch_img1', batch_img1, global_step)

        else:
            raise RuntimeError('Unknown training method: %s' % training_method)

        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1

        train_accuracy = n_correct / float(n_total)
        print('training accuracy: %.3f' % train_accuracy)
        vis.add_scalar('accuracy/train', train_accuracy, epoch * len(train_dataset))

        if validation_dataset is not None:
            validation_result = test_simple(nets, validation_dataset, vis, epoch * len(train_dataset), global_args)
            print('validation_accuracy: %.3f' % validation_result['accuracy'])
            vis.add_scalar('accuracy/validation', validation_result['accuracy'], epoch * len(train_dataset))
            vis.add_scalar('accuracy/validation-top5', validation_result['top5_accuracy'], epoch * len(train_dataset))

        if epoch % save_interval == 0 or epoch == max_epochs:
            print('saving model...')
            state = {
                'epoch': epoch,
                'global_args': global_args,
                'class_dict': train_dataset.class_dict,
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
