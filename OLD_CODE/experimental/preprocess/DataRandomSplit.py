import os
import random
dir_name = "/home/jifan/Dropbox/phytoliths"
images = os.listdir(dir_name + "/global_autocontrast_normalized")
first_class = []
second_class = []
for image in images:
    if image[0] == "b":
        first_class += [image]
    else:
        second_class += [image]
print len(first_class), len(second_class)

random.shuffle(first_class)
random.shuffle(second_class)
min_size = min((len(first_class), len(second_class)))
train = first_class[min_size*2//5:] + second_class[min_size*2//5:]
valid = first_class[min_size//5:min_size*2//5] + second_class[min_size//5:min_size*2//5]
test = first_class[:min_size//5] + second_class[:min_size//5]
only_train = train + valid
random.shuffle(train)
random.shuffle(valid)
random.shuffle(test)
random.shuffle(only_train)
'''
f = open(dir_name + "/train_images.txt", 'w')
for image in only_train:
    print >> f, image

f = open(dir_name + "/train_images_v.txt", 'w')
for image in train:
    print >> f, image

f = open(dir_name + "/valid_images_v.txt", 'w')
for image in valid:
    print >> f, image

f = open(dir_name + "/test_images.txt", 'w')
for image in test:
    print >> f, image
'''
print len(train), len(valid), len(test)
print "training size:", len(train)
print "test size:", len(test)
