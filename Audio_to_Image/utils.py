from __future__ import print_function
import sys
import os
import time
import logging
import re
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import h5py


#---------------------------for image----------------------------#
def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def read_list_file(file_name):
    with open(file_name, 'rb') as fp:
        content = fp.readlines()
    content = [data.strip().decode('utf8') for data in content]
    content = [data.replace(u"\uFFFD", " ") for data in content]
    content = [data.replace("  ", " ") for data in content]
        #content = fp.read().decode("utf8").split("\n")
    return content  #[x.strip() for x in content]  #delete space and \n

def read_dict_file(file_name):
    with open(file_name, 'rb') as fp:
        content = fp.readlines()
    dict_content = [x.strip().decode("utf8").split() for x in content]
    dict_content = {key:value for [key, value] in dict_content}
    return dict_content

# return dict, {class_id:class name}
def read_class_name(class_file_name):
    return read_dict_file(class_file_name)

# return a dict, {image_id:class_id}
def read_images_class_label(label_file_name):
    return read_dict_file(label_file_name)

def read_images_filename(image_list_filename):
    return read_dict_file(image_list_filename)

# return a list
def read_class_ids(filename):
    ids = read_list_file(filename)
    pattern = r"\b0*([1-9][0-9]*|0)"
    ids = [re.sub(pattern, r"\1", id) for id in ids]
    return ids

def read_bbox(filename):
    """read the bbox file
    return a dict:{ID:int tuple}
    """
    bbox = read_list_file(filename)
    pattern = re.compile(r"(\d+) (\d+.\d*) (\d+.\d*) (\d+.\d*) (\d+.\d*)")
    m_list = [pattern.match(bbox_line) for bbox_line in bbox]
    bbox_dict = {m.group(1):(float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))) for m in m_list}
    return bbox_dict 


# return a dict, length is the num of class, {class_id:filename_id_list}
def classify_images_by_class(class_dict, image_filename_dict, class_label_dict):
    class_filename_dict = { class_id:[] for class_id in class_dict.keys()}

    for (image_id, class_id) in class_label_dict.items():  #{image_id:class_id}
        class_filename_dict.get(class_id).append(image_filename_dict.get(image_id))
    return class_filename_dict

def classify_images_ids_by_class(class_dict, class_label_dict):
    class_filename_ids_dict = { class_id:[] for class_id in class_dict.keys()}

    for (image_id, class_id) in class_label_dict.items():  #{image_id:class_id}
        class_filename_ids_dict.get(class_id).append(image_id)
    return class_filename_ids_dict

def write_image_feature_h5(filename, data):
    with h5py.File(filename,'w') as fp:
        fp.create_dataset("image_feature", data=data)

def read_image_feature_h5(filename):
    with h5py.File(filename,'r') as fp:
        data = fp["image_feature"][:]
    return data

#----------------for text-------------------------#
def str2ascii(string_data):
    return [ord(s) for s in string_data]

def ascii2str(ascii_data):
    return [chr(a) for a in ascii_data]

def check_vocabulary(string_data, vocabulary):
    for c in string_data:
        if c not in vocabulary:
            vocabulary.append(c)
            print("add vocabulary:{}, from \"{}\"".format(c, string_data))

#data":[array, array]
def write_text_feature_h5(filename, data):
    fp = h5py.File(filename,'w')
    for idx in range(len(data)):
        fp.create_dataset(str(idx), data=data[idx])
    fp.close()

def read_text_feature_h5(filename):
    fp = h5py.File(filename,'r')
    data = []
    for idx in range(len(fp)):
        data.append(fp[str(idx)][:])
    fp.close()
    return data

def read_class_text_feature_h5(filename):
    with h5py.File(filename,'r') as fp:
        data = fp["text_feature"][:]
    return data
     
def write_class_text_feature_h5(filename, data):
    with h5py.File(filename,'w') as fp:
        data = fp.create_dataset("text_feature", data=data)


def read_text_idx_feature_h5(filename):
    with h5py.File(filename,'r') as fp:
        data = fp["text_idx_feature"][:]
    return data
     
def write_text_idx_feature_h5(filename, data):
    with h5py.File(filename,'w') as fp:
        data = fp.create_dataset("text_idx_feature", data=data)


def read_all_captions(filename):
    content = read_list_file(filename)
    captions = []
    for line in content:
        #line = line.replace("\ufffd", " ")
        string = line
        # clean data: reference: https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/data_loader_txt.py
        #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        #string = re.sub(r"\'s", " \'s", string)
        #string = re.sub(r"\'ve", " \'ve", string)
        #string = re.sub(r"n\'t", " n\'t", string)
        #string = re.sub(r"\'re", " \'re", string)
        #string = re.sub(r"\'d", " \'d", string)
        #string = re.sub(r"\'ll", " \'ll", string)
        #string = re.sub(r",", " , ", string)
        #string = re.sub(r"!", " ! ", string)
        #string = re.sub(r"\(", " \( ", string)
        #string = re.sub(r"\)", " \) ", string)
        #string = re.sub(r"\?", " \? ", string)
        string = re.sub(u"\ufffd", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        # clean data end
        line = string

        line = line.lower()
        if len(line)>0:
            captions.append(line)

    return captions

def read_text_vocabulary_json(text_feature_dir, default_flag=False):
    import json
    if default_flag:
        vocabulary_json_filename = os.path.join(text_feature_dir, "alphabet_default.json")
    else:
        vocabulary_json_filename = os.path.join(text_feature_dir, "alphabet.json")
    with open(vocabulary_json_filename, 'r') as f:
         vocabulary = json.load(f)
    return vocabulary

def write_text_vocabulary_json(text_feature_dir, vocabulary):
    import json
    vocabulary_json_filename = os.path.join(text_feature_dir, "alphabet.json")
    with open(vocabulary_json_filename, 'w') as f:
        json.dump(vocabulary, f)


def show_progress_bar(percentage):
    if percentage<0:
        print("show_progress_bar: parameter error")
        percentage = 0
    elif percentage>100:
        print("show_progress_bar: parameter error")
        percentage = 100
    percentage = int(percentage*100)

    s1 = "\r[{}{}]{}%".format(">"*percentage, " "*(100-percentage), percentage)
    if percentage==100:
        s1 += "\n"
    sys.stdout.write(s1)
    sys.stdout.flush()

def show_progress_bar_increment(inc, L=[]):
    if len(L)==0:
        L.append(0)
    L[0] += inc
    show_progress_bar(L[0])
    return L[0]


def convert_to_onehot(data, n_class):
    onehot = np.eye(n_class)[data.reshape(-1)]
    return onehot.reshape(list(data.shape)+[n_class])


def convert_one_sentence(text, vocabulary, target_len):
    # text_ascii = text.encode('ascii', 'ignore')
    text_ascii = str2ascii(text)
    text_idx_np = np.zeros((target_len,), dtype=np.int16)
    try:
        text_idx_np_temp = np.array([vocabulary.index(c)+1 for c in text])
        if text_idx_np_temp.shape[0]>target_len:
            text_idx_np_temp = text_idx_np_temp[:target_len]
        text_idx_np[:text_idx_np_temp.shape[0]] = text_idx_np_temp
    except ValueError:
        print("convert error: {}".format(text))
        # return None
    # text_onehot = convert_to_onehot(text_idx_np, len(vocabulary)+1)
    return text_idx_np  # text_onehot


def rename_by_order(full_name_src):
    if not os.path.exists(full_name_src):
        return full_name_src
    filename, ext = os.path.splitext(full_name_src)
    for idx in range(99999):
        full_name = filename+".{:05d}".format(idx)+ext
        if not os.path.exists(full_name):
            os.rename(src=full_name_src, dst=full_name)
            break
        else:
            continue
    return full_name

def get_ordered_name(full_name_base):
    filename, ext = os.path.splitext(full_name_base)
    for idx in range(99999):
        full_name = filename+"{:05d}".format(idx)+ext
        if not os.path.exists(full_name):
            os.path.isdir(full_name)
            os.makedirs(full_name)
            break
        else:
            continue
    return full_name



def sents2idx(sent_list, vocabulary, max_length=201):
    """
    return: array, (max_length, len_sent)
    """
    captions = sent_list
    #convert to ascii
    captions_ascii = [str2ascii(string_data) for string_data in captions]
    #save to h5 file
    captions_ascii_np = [np.array(caption_ascii) for caption_ascii in captions_ascii]
    #save idx to file
    captions_idx_np = np.zeros((max_length, len(captions_ascii_np)), dtype=np.int16)
    for idx in range(len(captions_ascii_np)):
        caption_idx_np = np.array([vocabulary.index(c)+1 for c in ascii2str(captions_ascii_np[idx])])
        if caption_idx_np.shape[0]>max_length:
            print("[text feature]: warning: clip the data")
            caption_idx_np = caption_idx_np[0:max_length]
        captions_idx_np[0:caption_idx_np.shape[0], idx] = caption_idx_np
    return captions_ascii_np, captions_idx_np


import pickle
def read_pickle(filepath, encoding="utf8"):
    with open(filepath, "rb") as fp:
        data = pickle.load(fp, encoding=encoding)
    return data


import scipy
import librosa

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


def load_one_audio_file(path, audio_conf, windows={'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}):
    audio_type = audio_conf.get('audio_type', 'melspectrogram')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
    preemph_coef = audio_conf.get('preemph_coef', 0.97)
    sample_rate = audio_conf.get('sample_rate', 16000)
    window_size = audio_conf.get('window_size', 0.025)
    window_stride = audio_conf.get('window_stride', 0.01)
    window_type = audio_conf.get('window_type', 'hamming')
    num_mel_bins = audio_conf.get('num_mel_bins', 40)
    target_length = audio_conf.get('target_length', 2048)
    use_raw_length = audio_conf.get('use_raw_length', False)
    padval = audio_conf.get('padval', 0)
    fmin = audio_conf.get('fmin', 20)
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)
    # load audio, subtract DC, preemphasis
    # print("load audio file:{}".format(path))
    y, sr = librosa.load(path, sample_rate)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    # compute mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=windows.get(window_type, windows['hamming']))
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    n_frames = logspec.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
            constant_values=(padval, padval))
    elif p < 0:
        logspec = logspec[:,0:p]
        n_frames = target_length

    return logspec, n_frames