from __future__ import division, print_function, absolute_import
import numpy as np 
from PIL import Image
import os
import json
import numpy

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
    return np.asarray(pil_image, dtype="float32").reshape(64, 64, 3)


def create_Resnet(num_classes):
    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 9
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


def predict(network, modelfile, images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == '__main__':
    #length = []
    net = create_Resnet(61)
    #folder_path = 'old_dataset/AgriculturalDisease_validationset/'
    #folderlist = os.listdir(folder_path)
    #for folder in folderlist:

    #path = folder_path + folder
    folder = "images3"
    path = 'test_A/' + folder
    imgs = []
    image_id = []

    result = []
    submit = './' + folder + '.json'
    filelist = os.listdir(path)
    for files in filelist:
        img_path = os.path.join(path, files)
        img = load_image(img_path)
        img = resize_image(img, 64, 64)
        imgs.append(pil_to_nparray(img))
        image_id.append(files)
    predicted_disease_class = predict(net, 'model_resnet-22656', imgs)
    disease_class = np.argmax(predicted_disease_class, 1)
    #print(image_id)

    #print(disease_class)
    for i in range(len(image_id)):
        temp_dict = {}
        temp_dict['image_id'] = image_id[i]
        #print(image_id[i])
        temp_dict['disease_class'] = disease_class[i]
        #print(disease_class[i])
        result.append(temp_dict)
    #length.append(len(image_id))
    with open(submit, 'w') as f:
        json.dump(result, f, cls=MyEncoder)
    #print(sum(length))
