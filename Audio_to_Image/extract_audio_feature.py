"""
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import scipy.signal
import torch
import tqdm

sys.path.append(os.getcwd())
from speech_encoder import CNNRNN
from utils import load_one_audio_file
from trainer import load_checkpoint



windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

@torch.no_grad()
def extract_one_feature(model, audio_file_paths, cuda=True):
    global windows
    audios_lens = [load_one_audio_file(audio_file_path, {}, windows) for audio_file_path in audio_file_paths]
    audios = [item[0] for item in audios_lens]
    lens = [item[1] for item in audios_lens]
    audio = np.array(audios)
    lens_np = np.array(lens)
    audio = torch.from_numpy(audio).float().squeeze()
    lens = torch.from_numpy(lens_np).long().squeeze()
    if len(audio.shape)<2:
        audio = audio.unsqueeze(0)
        # lens = lens.unsqueeze(0)

    # sort the data
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(lens, 0, True)
    sorted_data = audio[sorted_cap_indices]
    recover_index = torch.LongTensor([(sorted_cap_indices==idx).nonzero() for idx in range(len(sorted_cap_indices))])
    if cuda:
        sorted_data = sorted_data.float().cuda()
        sorted_cap_lens = sorted_cap_lens.long().cuda()
    # print(sorted_cap_indices, recover_index, sorted_data.shape, sorted_cap_lens.shape)
    if sorted_cap_lens[-1] < 64:
        sorted_cap_lens[-1] = sorted_cap_lens[-2]
        sorted_data[-1] = sorted_data[-2]
        print("data is too short, drop: ")
    with torch.no_grad():
        sorted_cap_lens //= 64
        feature = model.extract_feature(sorted_data, sorted_cap_lens)
        feature = feature[recover_index]
    
    return feature.detach().cpu().numpy(), lens_np


@torch.no_grad()
def extract_feature_for_StackGANv2(model, 
    train_json_file, val_json_file, train_filename_file, val_filename_file, feature_len=1024, audio_switch=0, cuda=True,
    extract_func=extract_one_feature):
    root = os.getcwd()
    with open(os.path.join(root, train_json_file), "r") as fp:
        train_json = json.load(fp)
    with open(os.path.join(root, val_json_file), "r") as fp:
        val_json = json.load(fp)

    with open(os.path.join(root, train_filename_file), "rb") as fp:
        train_filename = pickle.load(fp, encoding="bytes")
    with open(os.path.join(root, val_filename_file), "rb") as fp:
        val_filename = pickle.load(fp, encoding="bytes")

    for json_data, filename_list, filename_file in zip((train_json, val_json),
                                                    (train_filename, val_filename), 
                                                    (train_filename_file, val_filename_file)):
        audio_base_path = json_data["a_audio_base_path"]
        audio_feature_all = []
        audio_lens_all = []
        bar = tqdm.tqdm(filename_list)
        if not os.path.isdir(os.path.join(audio_base_path)):
            print("error: audio switch is no exist")
            return
        audio_file_paths = []
        for filename in bar:
            audio_file_paths = [os.path.join(audio_base_path, filename+"_{}.wav".format(_idx)) for _idx in range(10)]

            feature, lens = extract_func(model, audio_file_paths, cuda)
            audio_feature_all.append(feature)
            audio_lens_all.append(lens)
        bar.close()
        with open(os.path.join(os.path.dirname(filename_file), "audio_features_{}.pickle".format(audio_switch)), "wb") as fp:
            pickle.dump(audio_feature_all, fp)
        with open(os.path.join(os.path.dirname(filename_file), "audio_features_lens-{}.pickle".format(audio_switch)), "wb") as fp:
            pickle.dump(audio_lens_all, fp)



def extract_feature_for_StackGANv2_cnn_rnn_attn(model_file, train_json_file, val_json_file, 
    train_filename_file, val_filename_file, 
    class_num=150, embedding_dim=1024, dropout=0,
    feature_len=1024, audio_switch=0, cuda=True, **kwargs):
    root = os.getcwd()
    model = CNNRNN(40, embedding_dim=1024, nhidden=1024, nsent=1024, **kwargs)
    load_checkpoint(model, model_file, map_location=torch.device('cpu'), strict=True)
    if cuda:
        model = model.cuda()
    extract_feature_for_StackGANv2(
        model, train_json_file, val_json_file, train_filename_file, val_filename_file, feature_len, audio_switch
    )


import math
@torch.no_grad()
def extract_audio_feature(model, filenames):
    model = model.cuda()
    audio_features, audio_lens = [], []
    print("file num:", len(filenames))
    chunk_num = math.ceil(len(filenames)/10.0)
    for chunk_idx in tqdm.tqdm(list(range(chunk_num))):
        start_idx = chunk_idx * 10
        end_idx = min(start_idx + 10, len(filenames))
        filenames_temp = filenames[start_idx:end_idx]
        audio_features_temp, audio_lens_temp = extract_one_feature(model, filenames_temp)
        audio_features.append(audio_features_temp)
        audio_lens.append(audio_lens_temp)
    audio_features, audio_lens = np.concatenate(audio_features), np.concatenate(audio_lens)
    return audio_features, audio_lens



def extract_audio_feature_birds(model, audio_switch, **kwargs):
    root = os.getcwd()
    for split in ('train', 'val'):
        split_json = json.load(open(os.path.join(root, "./data/birds/{}.json".format(split)), 'r'))
        split_filenames = []
        for _d in split_json['data']:
            split_filenames += [os.path.join(split_json['a_audio_base_path'], __d) for __d in _d['audio']]
        
        
        split_features, split_audio_lens = extract_audio_feature(model, split_filenames)
        split_shape = split_features.shape
        split_features = split_features.reshape(split_shape[0]//10, 10, split_shape[1])
        split_audio_lens = split_audio_lens.reshape(split_shape[0]//10, 10)
        # save
        with open(os.path.join(root, "./data/birds/{}/audio_features_{}.pickle".format(split if split=="train" else "test", audio_switch)), 'wb') as fp:
            pickle.dump(split_features, fp)
        with open(os.path.join(root, "./data/birds/{}/audio_features_lens_{}.pickle".format(split if split=="train" else "test", audio_switch)), 'wb') as fp:
            pickle.dump(split_audio_lens, fp)
    

def extract_audio_feature_flowers(model, audio_switch, **kwargs):
    root = os.getcwd()
    for split in ('train', 'test'):
        split_json = json.load(open(os.path.join(root, "./data/flowers/{}.json".format(split)), 'r'))
        split_filenames = []
        for _d in split_json['data']:
            split_filenames += [os.path.join(split_json['audio_base_path'], __d) for __d in _d['wav']]
        split_features, split_audio_lens = extract_audio_feature(model, split_filenames)
        split_shape = split_features.shape
        split_features = split_features.reshape(split_shape[0]//10, 10, split_shape[1])
        split_audio_lens = split_audio_lens.reshape(split_shape[0]//10, 10)
        with open(os.path.join(root, "./data/flowers/{}_audio_features_{}.pickle".format(split, audio_switch)), "wb") as fp:
            pickle.dump(split_features, fp)
        with open(os.path.join(root, "./data/flowers/{}_audio_features_lens_{}.pickle".format(split, audio_switch)), "wb") as fp:
            pickle.dump(split_audio_lens, fp)

    

def extract_audio_feature_places(model_file, audio_switch, **kwargs):
    root = os.getcwd()
    audio_folder = os.path.join(root, "./data/Places_subset/audios")
    for split in ('train', 'val'):
        split_json = json.load(open(os.path.join(root, "./data/Places_subset/metadata/{}_split.json".format(split)), 'r'))
        split_filenames = []
        for _d in split_json['data']:
            split_filenames.append(os.path.join(audio_folder, _d['wav'][5:]))
        split_features, split_audio_lens = extract_audio_feature(model, split_filenames)
        print(split_features.shape)
        with open(os.path.join(root, "./data/Places_subset/metadata/{}_split_audio_features_{}.pickle".format(split, audio_switch)), "wb") as fp:
            pickle.dump(split_features, fp)
        with open(os.path.join(root, "./data/Places_subset/metadata/{}_split_audio_features_lens_{}.pickle".format(split, audio_switch)), "wb") as fp:
            pickle.dump(split_audio_lens, fp)



def get_parser():
    parser = argparse.ArgumentParser(description="extract_audio_feature")
    parser.add_argument("--train_json", type=str, default="./data/birds/train.json",
            help="train json file, default ./data/birds/train.json")
    parser.add_argument("--val_json", type=str, default="./data/birds/val.json",
            help="val json file, default ./data/birds/val.json")
    parser.add_argument("--model", type=str, default="./Audio_to_Image/model/model_best.pt",
            help="model parameters for AudioEncoder, default ./Audio_to_Image/model/AudioEncoder_best.pt")
    parser.add_argument("--audio_switch", type=str, default="0",
            help="audio swtich to be extracted")        
    parser.add_argument("--no_cuda", action="store_true", default=False, 
            help="disable cuda if set this value")
    parser.add_argument("--feature_len", type=int, default=1024, help="audio featuer length, 1024 or 2048, default 1024")        
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--dataset", choices=['birds', 'flowers', 'places'], default='birds', help="")
    parser.add_argument("--bidirectional", action='store_true', default=False, help="")
    
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
    

if __name__=="__main__":
    args = get_parser()
    print(args)
    model = CNNRNN(40, embedding_dim=1024, nhidden=1024, nsent=1024, bidirectional=args.bidirectional)
    load_checkpoint(model, args.model, map_location=torch.device('cpu'), strict=True)
    model.eval()
    if args.dataset == "birds":
        extract_audio_feature_birds(model, args.audio_switch)    
    elif args.dataset == "flowers":
        extract_audio_feature_flowers(model, args.audio_switch)
    elif args.dataset == "places":
        extract_audio_feature_places(model, args.audio_switch)
    else:
        raise NotImplementedError

