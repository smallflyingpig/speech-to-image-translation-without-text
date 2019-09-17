#!/usr/bin/python
# python2.7 with caffe
# referance : https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb
# use caffe to extract feature of all images

from __future__ import print_function
import sys
import os
import time
import multiprocessing
import logging
import re
import argparse
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import glob

sys.path.append(os.getcwd())

from utils import write_image_feature_h5, read_class_name, read_images_class_label, read_images_filename, \
        classify_images_by_class, check_dir, read_text_vocabulary_json, write_text_vocabulary_json, read_all_captions, \
        read_all_captions, check_vocabulary, str2ascii, write_text_feature_h5, write_class_text_feature_h5, read_class_text_feature_h5, \
        ascii2str, write_text_idx_feature_h5

from aip import AipSpeech
#
## display plots in this notebook
#
## set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



project_root = os.getcwd()




class BaiduSpeech(object):
    def __init__(self, APP_ID, API_KEY, SECRET_KEY, mode='tts'):
        call_dict = {'tts':self.tts, 'asr':self.asr}
        self.client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        self.client.setConnectionTimeoutInMillis(10000)
        self.client.setSocketTimeoutInMillis(10000)
        self.wait_time = 0.001
        self.call_func = call_dict[mode]
    
    @staticmethod
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def tts(self, texts, param={}):
        audios = []
        for text in texts:
            wait_time = self.wait_time
            while True:
                time.sleep(wait_time)
                try:
                    audio = self.client.synthesis(
                        text, "zh", 1, param
                        )
                    if not isinstance(audio, dict):
                        audios.append(audio)
                        break
                    else: # error
                        print("return error:{}".format(audio))
                        continue
                except Exception as e:
                    wait_time *= 2
                    print("Exception occur:{}".format(e))
                    continue
    
        return audios

    def asr(self, audio_file, param={}):
        wait_time = self.wait_time
        audio_format = param.pop('format', 'wav')
        audio_rate = param.pop('rate', 16000)
        # dev_pid = param.pop('dev_pid', 1737)

        # read wav
        audio_data = self.get_file_content(audio_file)
        text = None
        while True:
            time.sleep(wait_time)
            try:
                text_json = self.client.asr(
                    speech=audio_data, format=audio_format, rate=audio_rate, options=param
                    )
                if not isinstance(text_json, dict):
                    if text_json.get('err_no', 0) == 0:
                        text = text_json.get('result', '')
                        break
                    else:
                        print("error occur", text_json)
                else: # error
                    print("return error:{}".format(text_json))
                    continue
            except Exception as e:
                wait_time *= 2
                print("Exception occur:{}".format(e))
                continue
            return text
                
            
    def __call__(self, data, param={}):
        return self.call_func(data, param)


    
def get_bird_text_audio_path_pair(args):
    audio_dir = args.output_dir
    audio_dir = os.path.join(audio_dir, "audio")
    audio_dir = os.path.join(audio_dir, str(args.person))
    pairs_all = []
    for split in ('train', 'test'):
        with open(os.path.join(project_root, "./data/birds/{}.json".format(split)), 'r') as fp:
            json_data = json.load(fp)
        for _d in json_data['data']:
            captions = _d['text']
            audio_filenames = [os.path.join(audio_dir, filename) for filename in _d['audio']]
            pairs_all.append((captions, audio_filenames[0][:-6]))
    return pairs_all


def get_flower_text_audio_path_pair(args):
    # text_data_dir = os.path.join(project_root, "./data/flowers/cvpr2016_flowers/text_c10")
    audio_dir = args.output_dir
    pairs_all = []
    for split in ('train', 'test'):
        with open(os.path.join(project_root, "./data/flowers/{}.json".format(split)), 'r') as fp:
            json_data = json.load(fp)
        for _d in json_data['data']:
            captions = _d['text']
            audio_filenames = [os.path.join(audio_dir, filename) for filename in _d['wav']]
            pairs_all.append((captions, audio_filenames[0][:-6]))
    return pairs_all
import json


def get_place_text_audio_path_pair(args):
    audio_dir = args.output_dir
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
    print("process places dataset, save audios to ", audio_dir)
    json_files = ["./data/Places_subset/metadata/train.json", "./data/Places_subset/metadata/val.json"]
    pairs_all = []
    for json_file in json_files:
        with open(json_file, 'r') as fp:
            json_file = json.load(fp)
        for data in json_file.get('data', []):
            text = data.get('asr_text')
            audio_path = data["wav"][5:-4]
            audio_path = os.path.join(audio_dir, audio_path)
            pairs_all.append([text, audio_path])
    return pairs_all


def caption_to_audio(pairs_all, APP_ID, API_KEY, SECRET_KEY, person='0'):
    tts = BaiduSpeech(APP_ID, API_KEY, SECRET_KEY, mode='tts')
    print("caption to audio: len:{}".format(len(pairs_all)))
    for idx, (captions, audio_filename) in enumerate(pairs_all):
        check_dir(os.path.dirname(audio_filename))
        # convert to audio
        audios = tts(captions, {"aue":6, "per":person})  # wav
        #save to file
        for idx2, audio in enumerate(audios):
            audio_filename_temp = audio_filename+"_{}.wav".format(idx2)
            with open(audio_filename_temp, "wb") as fp:
                fp.write(audio)
        print("[{}/{}] convert file: {}".format(idx, len(pairs_all), audio_filename))

def convert_a_text(text, save_path, APP_ID, API_KEY, SECRET_KEY, person='0'):
    tts = BaiduSpeech(APP_ID, API_KEY, SECRET_KEY, mode='tts')
    audio = tts([text], {"aue":6, "per":person})[0]
    with open(save_path, "wb") as fp:
        fp.write(audio)
    print("save to file:", save_path)

def caption_to_audio_one_by_one(pairs_all, APP_ID, API_KEY, SECRET_KEY, check_size_threshold=100, person='0'):
    tts = BaiduSpeech(APP_ID, API_KEY, SECRET_KEY, mode='tts')
    print("caption to audio: len:{}".format(len(pairs_all)))
    for idx, (captions, audio_filename) in enumerate(pairs_all):
        check_dir(os.path.dirname(audio_filename))
        if isinstance(captions, list):
            for idx2, caption in enumerate(captions):
                audio_filename_temp = audio_filename+"_{}.wav".format(idx2)
                if os.path.exists(audio_filename_temp) and os.path.getsize(audio_filename_temp)>check_size_threshold:
                    print("skip file:", audio_filename_temp)
                    continue
                else:
                    audio = tts([caption], {"aue":6, "per":person})[0]
                with open(audio_filename_temp, "wb") as fp:
                    fp.write(audio)
        else:
            caption = captions
            audio_filename_temp = audio_filename
            if len(audio_filename_temp)<4 or audio_filename_temp[-4:] != ".wav":
                audio_filename_temp += ".wav"
            if os.path.exists(audio_filename_temp) and os.path.getsize(audio_filename_temp)>check_size_threshold:
                print("skip file:", audio_filename_temp)
                continue
            else:
                audio = tts([caption], {"aue":6, "per":person})[0]
            with open(audio_filename_temp, "wb") as fp:
                fp.write(audio)
                
            
        print("[{}/{}] convert file: {}".format(idx, len(pairs_all), audio_filename))


def main(args):
    if args.text is not None and args.save_path is not None:
        convert_a_text(args.text, args.save_path, args.APP_ID, args.API_KEY, args.SECRET_KEY)
        return

    if args.dataset == "birds":
        pairs_all = get_bird_text_audio_path_pair(args)
    elif args.dataset == "flowers":
        pairs_all = get_flower_text_audio_path_pair(args)
    elif args.dataset == 'places':
        pairs_all = get_place_text_audio_path_pair(args)
    else:
        raise NotImplementedError
    if args.one_by_one:
        caption_to_audio_one_by_one(pairs_all, args.APP_ID, args.API_KEY, args.SECRET_KEY)
    else:
        caption_to_audio(pairs_all, args.APP_ID, args.API_KEY, args.SECRET_KEY)




def get_parser():
    parser = argparse.ArgumentParser(prog="Text to Speech")
    parser.add_argument("--APP_ID", type=str, help="APP_ID for baidu aip", required=True)
    parser.add_argument("--API_KEY", type=str, help="API_KEY for baidu aip", required=True)
    parser.add_argument("--SECRET_KEY", type=str, help="SECRET_KEY for baidu aip", required=True)
    parser.add_argument("--output_dir", type=str, default="./data/Places_subset/audios_0", 
        help="folder for audio, default ./data/birds/CUB_200_2011_audio")
    parser.add_argument("--person", type=int, default=0, 
        help="param per for Baidu aip, [0,1,3,4], default 0")
    parser.add_argument("--dataset", choices=['birds', 'flowers', 'places'], default="places", help="select the dataset, bird or flower")
    parser.add_argument("--one_by_one", action='store_true', default=False, help="")
    parser.add_argument("--text", type=str, default=None, help="")
    parser.add_argument("--save_path", type=str, default=None, help="")
    args, _ = parser.parse_known_args()
    return args


if __name__=="__main__":
    args = get_parser()
    main(args)
    
    


