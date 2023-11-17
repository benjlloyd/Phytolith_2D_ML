# THIS FILE IS NO LONGER COMPATIBLE WITH THE REST OF THE CODE.
# PLEASE MIGRATE.

# import torch
# from torch import nn
# from torch.optim import lr_scheduler
# import torch.optim as optim
# from train_fixture import TrainingFixture
# from metric_train_fixture import MetricTrainingFixture
# import datasets
# import models
#
#
# from args import g
#
# def create_datasets():
#     if g.random_rotation:
#         angles = [float(t) for t in g.angles.split(',')]
#     else:
#         angles = False
#
#     metric_train_dataset = datasets.PairDataset(
#         g.image_dir, g.train_list_file, g.granularity, g.image_size, random_rotation=angles)
#
#     linear_train_dataset = datasets.Dataset(
#         g.image_dir, g.train_list_file, g.granularity, g.image_size, random_rotation=angles)
#
#     class_dict = linear_train_dataset.class_dict
#
#     validation_dataset = datasets.Dataset(g.image_dir, g.validation_list_file, g.granularity, g.image_size,
#                                  class_dict)
#
#     test_dataset = datasets.Dataset(g.image_dir, g.test_list_file, g.granularity, g.image_size, class_dict)
#
#     print('%d training images' % len(linear_train_dataset))
#     print('%d validation images' % len(validation_dataset))
#     print('%d test images' % len(test_dataset))
#     print('%d classes' % len(class_dict))
#
#     return metric_train_dataset, linear_train_dataset, validation_dataset, test_dataset
#
#
# metric_train_dataset, linear_train_dataset, validation_dataset, test_dataset = create_datasets()
#
# feature_extractor = models.ResNet18Metric()
#
# metric_opt = optim.Adam(filter(lambda p: p.requires_grad, feature_extractor.parameters()), lr=g.metric_learning_rate,
#                         weight_decay=5e-5)
# metric_scheduler = lr_scheduler.StepLR(metric_opt, step_size=g.metric_lr_decay_steps, gamma=g.metric_lr_decay_rate)
#
# metric_crit = nn.HingeEmbeddingLoss(margin=100)
#
# metric_trainer = MetricTrainingFixture(feature_extractor, metric_opt, metric_crit, metric_scheduler)
#
# best_accuracy = 0
#
# for epoch in range(g.metric_max_epochs):
#     print('----- epoch %d -----' % epoch)
#
#     metric_trainer.net.train(False)
#
#     net = models.LinearNet(metric_trainer.net, len(linear_train_dataset.class_dict))
#
#     linear_opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=g.learning_rate, weight_decay=5e-5)
#     linear_scheduler = lr_scheduler.StepLR(linear_opt, step_size=g.lr_decay_steps, gamma=g.lr_decay_rate)
#
#     linear_crit = nn.CrossEntropyLoss()
#     linear_trainer = TrainingFixture(net, linear_opt, linear_crit, linear_scheduler)
#
#     linear_trainer.max_result = best_accuracy
#     linear_trainer.train_test(linear_train_dataset, test_dataset, g.max_epochs)
#     best_accuracy = max(best_accuracy, linear_trainer.max_result)
#     print "*"*5, 'best accuracy:', best_accuracy, "*"*5
#
#     metric_trainer.net.train(True)
#     metric_trainer.train_one_epoch(metric_train_dataset)
