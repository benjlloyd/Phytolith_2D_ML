# THIS FILE IS NO LONGER COMPATIBLE WITH THE REST OF THE CODE.
# PLEASE MIGRATE.

# import models
# import torch
# from torch.autograd import Variable
# import sys
# import numpy as np
#
#
# MODEL_FILE = sys.argv[1]
# IMAGE_FILE = sys.argv[2]
#
# sys.stderr.write('load model')
# state_dict = torch.load(MODEL_FILE)
# class_dict = state_dict['class_dict']
# class_list = sorted(class_dict.items(), key=lambda a: a[1])
#
# sys.stderr.write('construct net')
# feature_extractor = models.ResNet18Feature()
# net = models.LinearNet(feature_extractor, len(class_dict))
# sys.stderr.write('load state dict')
# net.load_state_dict(state_dict['net'])
# net.train(False)
#
#
# mean = np.array([0.485, 0.456, 0.406], np.float32).reshape((1, 3, 1, 1))
# std = np.array([0.229, 0.224, 0.225], np.float32).reshape((1, 3, 1, 1))
#
#
# from preprocess.normalization import resize_and_crop
# import cv2
# img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
# if img is None:
#     print 'failed to read %s.' % IMAGE_FILE
#     exit(0)
#
# img = img.astype(np.float32) / 255.0
# img = resize_and_crop(img, 224)
#
# img = np.stack([img, img, img], axis=0)
# img = img[None]
# img = (img - mean) / std
# img_v = Variable(torch.from_numpy(img), volatile=True)
#
# pred = net(img_v)[0].data.cpu().numpy()
#
# class_idx = np.argmax(pred)
# print 'most probable class: ', class_list[class_idx][0]
