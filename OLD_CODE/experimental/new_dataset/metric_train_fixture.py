# THIS FILE IS NO LONGER COMPATIBLE WITH THE REST OF THE CODE.
# PLEASE MIGRATE.

# from __future__ import print_function, division
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import numpy as np
# from torch.utils.data import DataLoader
# import gflags
# import sys
# import models
# import datasets
# import args
# from args import g
#
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
#
#
# class MetricTrainingFixture:
#     def __init__(self, net, opt, crit, lr_scheduler):
#         self.net = net
#         self.opt = opt
#         self.crit = crit
#         self.lr_scheduler = lr_scheduler
#
#     def train_one_epoch(self, dataset):
#         loader = DataLoader(dataset,
#                             batch_size=g.batch_size,
#                             shuffle=True,
#                             num_workers=1,
#                             pin_memory=True)
#
#         losses = []
#
#         for idx, (img_batch, sim_batch, dis_batch) in enumerate(loader):
#             img_batch_v = Variable(img_batch.cuda(async=True), requires_grad=False)
#             sim_batch_v = Variable(sim_batch.cuda(async=True), requires_grad=False)
#             dis_batch_v = Variable(dis_batch.cuda(async=True), requires_grad=False)
#
#             self.opt.zero_grad()
#
#             out_img = self.net(img_batch_v)
#             out_sim = self.net(sim_batch_v)
#             out_dis = self.net(dis_batch_v)
#
#             loss = self.crit(torch.squeeze(torch.norm(out_img - out_sim, 2, 1)),
#                              Variable(torch.ones(img_batch_v.size(0)).cuda())) + \
#                    self.crit(torch.squeeze(torch.norm(out_img - out_dis, 2, 1)),
#                              Variable(-torch.ones(img_batch_v.size(0)).cuda()))
#
#             loss.backward()
#             self.opt.step()
#
#             losses.append(loss.data[0])
#
#         avg_loss = np.mean(losses)
#         print('train loss: %.3f' % (avg_loss))
#
#     def evaluate(self, dataset):
#         '''
#         :param dataset: This should be a dataset that returns image and labels rather than a PairDataset
#         :return:
#         '''
#         loader = DataLoader(dataset,
#                             batch_size=g.batch_size,
#                             shuffle=False,
#                             num_workers=1,
#                             pin_memory=True)
#
#         self.net.train(False)
#
#         features = []
#
#         for idx, (img_batch, _) in enumerate(loader):
#             img_batch_v = Variable(img_batch.cuda(async=True), requires_grad=False)
#             out = self.net(img_batch_v)
#             features.append(out)
#
#         return torch.cat(features)
