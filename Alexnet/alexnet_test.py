from __future__ import division, print_function, absolute_import
import numpy as np 
from PIL import Image
import os
import json

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


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
    return np.asarray(pil_image, dtype="float32").reshape(224, 224, 3)


def create_alexnet(num_classes):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)
    return network


def predict(network, modelfile, images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)


if __name__ == '__main__':
    net = create_alexnet(61)
    path = 'AgriculturalDisease_testA\\images\\'
    imgs = []
    image_id = []
    temp_dict = {}
    result = []
    submit = './submit.json'
    filelist = os.listdir(path)
    for files in filelist:
        img_path = os.path.join(path, files)
        img = load_image(img_path)
        img = resize_image(img, 224, 224)
        imgs.append(pil_to_nparray(img))
        image_id.append(files)
    predicted_disease_class = predict(net, 'model_save.model', imgs)
    disease_class = np.argmax(predicted_disease_class, 1)
    for i in range(len(image_id)):
        temp_dict['image_id'] = image_id[i]
        temp_dict['disease_class'] = disease_class[i]
        result.append(temp_dict)
    with open(submit, 'w') as f:
        json.dump(result, f)


