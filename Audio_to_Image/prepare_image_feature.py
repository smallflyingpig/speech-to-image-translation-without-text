#!/usr/bin/python
# python2.7 with caffe, cuda 9.1
# referance : https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb
# use caffe to extract feature of all images
# use caffe docker to extract the feature

from __future__ import print_function
import argparse
import sys
import os
import time
import multiprocessing
import logging
import re
import json
import tqdm
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import pickle
sys.path.append(os.getcwd())
from utils import write_image_feature_h5, read_class_name, read_images_class_label, read_images_filename, \
        classify_images_by_class, check_dir, read_text_vocabulary_json, write_text_vocabulary_json, read_all_captions, \
        read_all_captions, check_vocabulary, str2ascii, write_text_feature_h5, write_class_text_feature_h5, read_class_text_feature_h5, \
        ascii2str, write_text_idx_feature_h5, read_class_ids, sents2idx, read_dict_file

#
## display plots in this notebook
#
## set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


parser = argparse.ArgumentParser(description="extract image feature")
parser.add_argument("--caffe_root", type=str, default="", help="")
parser.add_argument("--mean_file", type=str, default="", help="")
parser.add_argument("--dataset", choices=["birds", "flowers", "places"], default="birds", help="")
parser.add_argument('--task', type=str, default='zsl', 
                    help='zsl or classification')

args = parser.parse_args()
project_root = os.getcwd()

if not args.caffe_root == '':
    caffe_root = args.caffe_root
    if not args.mean_file == '':
        mean_file = args.mean_file
    else:
        mean_file = os.path.join(caffe_root, "python/caffe/imagenet/ilsvrc_2012_mean.npy")
else:
    caffe_root = "/home/ubuntu/caffe"
    mean_file = os.path.join(caffe_root, "python/caffe/imagenet/ilsvrc_2012_mean.npy")

print(caffe_root, mean_file)
sys.path.append(os.path.join(caffe_root, "python"))
import caffe

if os.path.isfile(os.path.join(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel')):
    print('CaffeNet found.')
else:
    print('Downloading pre-trained CaffeNet model...')
    #!../scripts/download_model_binary.py ../models/bvlc_googlenet
    os.system("python " + os.path.join(caffe_root, "scripts/download_model_binary.py ") + os.path.join(caffe_root, "models/bvlc_googlenet"))

caffe.set_mode_gpu()
caffe.set_device(0)


#----------------------------------------------------------------------------------------------------
#extract feature using GoogleLeNet, refer: Learning Deep Representations of Fine-Grained Visual Descriptions
'''
1. read filename
    2. for each file:
       2.1 load image
       2.2 resize to 227x227, clip and flip to 10 subimage(224x224)
       2.3 transformer to (C,H,W),[0,255],BGR, set mean
       2.4 net.forward and extract feature[pool5/7x7_s1]
       2.5 save feature to feature_dir
'''
def get_one_image_feature(net, transformer, image_filename, img_crop_xy, feature_layer_name):
    assert(transformer != None)
    img = caffe.io.load_image(image_filename)  #(H,W,C),RGB,[0,1]
    img = transformer.preprocess('data', img) #(C,H,W), BGR, [0,255], (3,224,224)
    img_flip = np.fliplr(img)
    imgs = [img[:, xy[1]:xy[3],xy[0]:xy[2]] for xy in img_crop_xy]
    imgs_flip = [img_flip[:, xy[1]:xy[3],xy[0]:xy[2]] for xy in img_crop_xy]
    imgs_all = imgs+imgs_flip  #10 view for a image
    for idx_temp in range(10):
        net.blobs['data'].data[idx_temp] = imgs_all[idx_temp]
    net.forward()
    features = net.blobs[feature_layer_name].data.copy()  #(10,1024,1,1) array, copy is important
    return np.transpose(np.squeeze(features),(1,0))  #(1024x10)



def load_net_transformer():
    model_def = os.path.join(caffe_root, 'models/bvlc_googlenet/deploy.prototxt')
    model_weights = os.path.join(caffe_root, 'models/bvlc_googlenet/bvlc_googlenet.caffemodel')
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    transform_shape = (10, 3, 227,227)
    feature_layer_name = "pool5/7x7_s1"
    mean_value = np.load(mean_file).mean(1).mean(1)
    # mean_value = np.expand_dims(mean_value, 1)
    # mean_value = np.expand_dims(mean_value, 2)
    print(mean_value.shape)
    #transformer for caffe
    transformer = caffe.io.Transformer({"data":transform_shape})
    transformer.set_transpose("data",(2,0,1))  #(H,W,C)==>(C,H,W)
    transformer.set_mean("data", mean_value)
    transformer.set_raw_scale("data",255)
    transformer.set_channel_swap("data",(2,1,0))  #RGB==>BGR
    img_crop_xy = [[0,0,224,224],[3,0,227,224],[1,1,225,225],[0,3,224,227],[3,3,227,227]]
    return net, transformer, img_crop_xy, feature_layer_name


def extract_birds_image_feature():
    train_json_file = "./data/birds/train.json"
    val_json_file = "./data/birds/val.json"
    net, transformer, img_crop_xy, feature_layer_name = load_net_transformer()
    for json_file in (train_json_file, val_json_file):
        with open(json_file, "r") as fp:
            json_data = json.load(fp)
        feature_all = []
        feature_filename = json_data['image_feature_path']
        image_base_path = json_data['image_base_path']
        data = json_data['data']
        bar = tqdm.tqdm(data)
        for item in bar:
            image_file = os.path.join(image_base_path, "images", item['image'])
            image_feature = get_one_image_feature(net, transformer, image_file, img_crop_xy, feature_layer_name)
            feature_all.append(image_feature.transpose())

        with open(feature_filename, "wb") as fp:
            pickle.dump(feature_all, fp)


def extract_flowers_image_feature():
    feature_path = "./data/flowers/"
    train_json_file = "./data/flowers/train.json"
    test_json_file = "./data/flowers/test.json"
    net, transformer, img_crop_xy, feature_layer_name = load_net_transformer()
    check_dir(feature_path)
    for json_file, feature_file in zip((train_json_file, test_json_file), ("train_image_feature_caffe.pickle", "test_image_feature_caffe.pickle")):
        json_data = json.load(open(json_file, "r"))
        feature_all = []
        feature_name = os.path.join(feature_path, feature_file)
        image_folder = json_data['image_base_path']
        data = json_data['data']
        bar = tqdm.tqdm(data)
        for item in bar:
            image_file_all = os.path.join(image_folder, item['img'])
            image_feature = get_one_image_feature(net, transformer, image_file_all, img_crop_xy, feature_layer_name)
            feature_all.append(image_feature.transpose())
        with open(feature_name, "wb") as fp:
            pickle.dump(feature_all, fp)


def extract_places_subset_image_feature():
    feature_path = "./data/Places_subset/metadata"
    train_json_file = "./data/Places_subset/metadata/train_split.json"
    val_json_file = "./data/Places_subset/metadata/val_split.json"

    net, transformer, img_crop_xy, feature_layer_name = load_net_transformer()

    for json_file, feature_file in zip( (train_json_file, val_json_file), ("train_split_caffe.pickle", "val_split_caffe.pickle") ):
        with open(json_file, "r") as fp:
            json_data = json.load(fp)
        feature_all = []
        check_dir(feature_path)
        feature_filename = os.path.join(feature_path, feature_file)
        # image_base_path = json_data['image_base_path']
        image_base_path = "./data/Places_subset/data"
        data = json_data['data']
        bar = tqdm.tqdm(data)
        for item in bar:
            image_file_all = os.path.join(image_base_path, item['image'][2:])
            image_feature = get_one_image_feature(net, transformer, image_file_all, img_crop_xy, feature_layer_name)
            feature_all.append(image_feature.transpose())
        
        with open(feature_filename, "wb") as fp:
            pickle.dump(feature_all, fp)






if __name__=="__main__":
    print(args)
    if args.dataset == "birds":
        extract_birds_image_feature()
    elif args.dataset == "flowers":
        extract_flowers_image_feature()
    elif args.dataset == "places":
        extract_places_subset_image_feature()
    else:
        raise NotImplementedError

    # extract_places_image_feature()
    