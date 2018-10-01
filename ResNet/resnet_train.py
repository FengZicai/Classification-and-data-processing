from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
from PIL import Image
import json

import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



# Data loading

# from tflearn.datasets import cifar10
# (X, Y), (testX, testY) = cifar10.load_data()
# Y = tflearn.data_utils.to_categorical(Y)
# testY = tflearn.data_utils.to_categorical(testY)


def load_image(img_path):
    img = Image.open(img_path)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def load_data(path, datafile, num_clss, save=False, save_path='dataset.pkl'):
    with open(datafile, 'r') as f:
        train_list = json.load(f)
    labels = []
    images = []
    for line in train_list:
        fpath = path +line['image_id']
        print(fpath)
        img = load_image(fpath)
        img = resize_image(img, 64, 64)
        np_img = pil_to_nparray(img)
        images.append(np_img)

        index = int(line['disease_class'])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels

def load_data2(path, datafile, num_clss, save=False, save_path='dataset.pkl'):
    with open(datafile, 'r') as f:
        train_list = json.load(f)
    labels = []
    images = []
    for line in train_list:
        fpath = path + line['disease_class'] + '/' + line['image_id']
        print(fpath)
        img = load_image(fpath)
        img = resize_image(img, 64, 64)
        np_img = pil_to_nparray(img)
        images.append(np_img)

        index = int(line['disease_class'])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X,Y


# # Real-time data preprocessing
# img_prep = tflearn.ImagePreprocessing()
# img_prep.add_featurewise_zero_center(per_channel=True)
#
# # Real-time data augmentation
# img_aug = tflearn.ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_crop([224, 224], padding=4)


# Building Residual Network
def create_Resnet(num_classes):
    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 12
    network = tflearn.input_data(shape=[None, 64, 64, 3])#, data_preprocessing=img_prep, data_augmentation=img_aug)
    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    network = tflearn.residual_block(network, n-1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    network = tflearn.residual_block(network, n-1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    # Regression
    network = tflearn.fully_connected(network, num_classes, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=mom,
                             loss='categorical_crossentropy')
    return network


# Training
def train(net, X, Y, testX, testY):
    model = tflearn.DNN(net, checkpoint_path='model_resnet',
                        max_checkpoints=5, tensorboard_verbose=0, tensorboard_dir='output',
                        clip_gradients=0.)
    if os.path.isfile('model_save.model.meta'):
        print("loading the model")
        model.load("model_save.model")
    elif os.path.isfile('model_resnet-14098.meta'):
        print('loading the model')
        try:
            model.load('model_resnet-14098')
        except Exception as e:
            print(e)
            pass
    else:
        print("NO file to load,error")
        pass
    model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
              snapshot_epoch=10, snapshot_step=False,
              show_metric=True, batch_size=128, shuffle=True,
              run_id="residual_train")


if __name__ == '__main__':
    # train_path = 'train_change/'
    # validation_path = 'AgriculturalDisease_validationset_change/images/'
    # X, Y = load_data2(train_path, 'train.json', 61, True, save_path='trainset.pkl')
    # testX, testY = load_data(validation_path, 'AgriculturalDisease_validation_annotations.json', 61, True, save_path='validationset.pkl')
    X, Y = load_from_pkl('trainset.pkl')
    testX, testY = load_from_pkl('validationset.pkl')
    # print(len(Y))
    # Y = np.asarray(Y, dtype='int32')
    # print(np.max(Y) + 1)
    #Y = tflearn.data_utils.to_categorical(Y, 61)

    # testY = np.asarray(testY, dtype='int32')
    # print(np.max(testY)+1)
    #testY = tflearn.data_utils.to_categorical(testY, 61)
    net = create_Resnet(61)
    train(net, X, Y, testX, testY)
