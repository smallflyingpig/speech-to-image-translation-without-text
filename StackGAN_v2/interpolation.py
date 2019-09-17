"""
1. input a text, TTS to a audio
2. extract feature using a pretrained model
3. generate an image using a pretrained generator
"""



import argparse
import os
import os.path as osp
import numpy as np 
from PIL import Image

import torch

from Audio_to_Image.models.cnn_rnn import CNNRNN_Attn
from Audio_to_Image.text_to_speech import caption_to_audio_one_by_one
from Audio_to_Image.trainer import load_checkpoint
from Audio_to_Image.extract_audio_feature import extract_only_one_feature

from StackGAN_v2.model import G_NET
from StackGAN_v2.miscc.config import cfg_from_file, cfg


def get_parser():
    parser = argparse.ArgumentParser("interpolation")
    parser.add_argument("--input_text", type=str, default="this flower is yellow in color", help="")
    parser.add_argument("--input_text_other", type=str, default="this flower is white in color", help="")
    parser.add_argument("--audio_encoder_path", type=str, 
        default="./output/Audio_to_Image/log/audio_encoder_flowers_cnnrnn_no_attn_1_0.1_googlenet_caffe/epoch_300.pth",
        help="path for pretrained audio encoder")
    parser.add_argument("--gan_path", type=str, 
        default="./output/StackGAN_v2/flowers_audio_feature_googlenet/netG_305000.pth",
        help="path for pretrained generator")
    parser.add_argument("--out_rnn_layer", action="store_true", default=False, help="")
    parser.add_argument("--out_attn_layer", action="store_true", default=False, help="")
    parser.add_argument("--gan_cfg", type=str, default="./StackGAN_v2/cfg/eval_birds.yml", help="")
    parser.add_argument("--output_dir", type=str, default="./output/interpolation", help="")
    parser.add_argument("--interpolation_num", type=int, default=10, help="")
    parser.add_argument("--img_filename_base", type=str, default="img", help="")
    parser.add_argument("--type", choices=['inference', 'interpolation'], default='inference')

    args = parser.parse_args()
    return args

def save_image(image, filename):
    image = image.squeeze()
    img = image.add(1).div(2).mul(255).clamp(0, 255).byte()
    ndarr = img.permute(1, 2, 0).data.cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def interpolate(audio_encoder, generator, args, interpolate_num=10, img_filename_base='img'):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    audio_temp_filename = (osp.join(args.output_dir, "audio0_{}.wav".format(img_filename_base)), 
        osp.join(args.output_dir, "audio1_{}.wav".format(img_filename_base)))
    print("text to speech")
    pairs = [(args.input_text, audio_temp_filename[0]),
        (args.input_text_other, audio_temp_filename[1])]
    caption_to_audio_one_by_one(pairs)
    print("extract audio feature...")
    audio_feature_all, feature_len_all = [], []
    for audio_filename in audio_temp_filename:
        audio_features, feature_lens = extract_only_one_feature(audio_encoder, audio_filename, cuda=False)
        audio_feature_all.append(audio_features[0])
        feature_len_all.append(feature_lens[0])
    for idx in range(interpolate_num+1):
        alpha = float(idx)/(interpolate_num)
        audio_feature_all.append(audio_feature_all[0]*alpha+audio_feature_all[1]*(1-alpha))

    audio_feature_all = torch.from_numpy(np.array(audio_feature_all[2:]))
    noise = torch.FloatTensor(1, cfg.GAN.Z_DIM).normal_(0,1)
    if cfg.CUDA:
        noise = noise.cuda()
        audio_feature_all = audio_feature_all.cuda()
        generator = generator.cuda()
    else:
        generator = generator.cpu()
    print("generate images")
    for idx, audio_feature in enumerate(audio_feature_all):
        # noise = torch.FloatTensor(1, cfg.GAN.Z_DIM).normal_(0,1)
        # if cfg.CUDA:
        #     noise = noise.cuda()
        audio_feature = audio_feature.float().unsqueeze(0)
        imgs, _, _ = generator.forward(noise, audio_feature)
        save_image(imgs[-1], osp.join(args.output_dir, img_filename_base+"_{}.jpg".format(idx)))
    
    
def inference(audio_encoder, generator, args):
    # get audio filename
    # inference
    audio_temp_path = osp.join(args.output_dir, "./audio_test.wav")
    image_temp_path = osp.join(args.output_dir, "./image_test.jpg")
    print("text to speech")
    pairs = [(args.input_text, audio_temp_path)]
    caption_to_audio_one_by_one(pairs)

    print("extract audio feature...")
    audio_features, feature_lens = extract_only_one_feature(audio_encoder, audio_temp_path, cuda=False)
    audio_feature, feature_len = audio_features[0], feature_lens[0]

    print("generate image...")
    
    noise = torch.FloatTensor(1, cfg.GAN.Z_DIM).normal_(0,1)
    audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0)
    if cfg.CUDA:
        noise = noise.cuda()
        audio_feature = audio_feature.cuda()
        generator = generator.cuda()
    else:
        generator = generator.cpu()
    imgs, _, _ = generator.forward(noise, audio_feature)
    save_image(imgs[-1], image_temp_path)
    print("save image to ", image_temp_path)

def main(args):
    # load models
    print("load model...")
    audio_encoder = CNNRNN_Attn(40, embedding_dim=1024, nhidden=1024, nsent=1024, attn_layer=args.out_attn_layer, out_rnn_layer=args.out_rnn_layer)
    load_checkpoint(audio_encoder, args.audio_encoder_path, map_location=torch.device('cpu'), strict=True)
    audio_encoder.eval().cpu()    

    cfg_from_file(args.gan_cfg)
    generator = G_NET()
    load_checkpoint(generator, args.gan_path, map_location=torch.device('cpu'), strict=True)
    generator.eval().cpu()
    
    if args.type == 'inference':
        inference(audio_encoder, generator, args)
    elif args.type == 'interpolation':
        interpolate(audio_encoder, generator, args, interpolate_num=args.interpolation_num, img_filename_base=args.img_filename_base)
    else:
        raise NotImplementedError


if __name__=="__main__":
    args = get_parser()
    main(args)




    

