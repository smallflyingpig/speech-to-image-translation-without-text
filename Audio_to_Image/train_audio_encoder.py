"""baseline model
1. Deep Cross-Modal Audio-Visual Generation
2. CMCGAN: A Uniform Framework for Cross-Modal Visual-Audio Mutual Generation
"""
import os
import h5py
import argparse
import logging
import time
import tqdm
import json
import pickle
import random
import numpy as np 

import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F 
from torch.utils.data  import Dataset, DataLoader
import torch.distributed as dist
from trainer import ClassifierTrainer
from speech_encoder import CNNRNN
from jel import JointEmbeddingLossLayer
from utils import convert_one_sentence, convert_to_onehot, load_one_audio_file
torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

class DatasetBase(Dataset):
    def __init__(self):
        pass

    @staticmethod
    def encode_text(text):
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ",", ";", ".", "!", "?", ":", "'", "\"", "\\", "/", "|", "_", "@", "#", "$", "%", "^", "&", "*", "~", "`", "+", "-", "=", "<", ">", "(", ")", "[", "]", "{", "}", "\n", " "]
        text_idx = convert_one_sentence(text, alphabet, target_len=201)
        text_onehot = convert_to_onehot(data=text_idx-1, n_class=len(alphabet))
        return text_onehot
    
    @staticmethod
    def get_rand(feature):
        # feature: (10, 1024)
        rand_idx = random.randint(0, len(feature)-1)
        return feature[rand_idx], rand_idx

class PlaceSubSet(DatasetBase):
    class_label = {
        'bedroom':0, 'dinette':1, 'dining_room':2, 'home_office':3, 'hotel_room':4,
        'kitchenette':5, 'living_room':6
    }
    def __init__(self, data_root, train=True):
        # read meta data
        split = "train" if train else "test"
        self.json_data = json.load(open(os.path.join(data_root, "{}.json".format(split)), "r"))
        self.image_folder = self.json_data['image_base_path']
        self.audio_folder = self.json_data['audio_base_path']
        # load image embedding
        image_feature_path = self.json_data['image_feature_path']
        self.image_embedding = pickle.load(open(image_feature_path, mode='rb'), encoding='latin1')
        print("load image feature from: {}".format(image_feature_path))
        self.data = self.json_data['data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        json_data = self.data[index]
        image_path = json_data["image"][2:]
        audio_path = json_data['wav'][5:]
        audio_data, audio_len = load_one_audio_file(
            os.path.join(self.audio_folder, audio_path), 
            {"target_length":2048}
            )
        audio_data=audio_data/np.max(np.abs(audio_data)) # norm the data
        rand_idx_image = random.randint(0, 9)
        image_data = self.image_embedding[index][rand_idx_image]
        text = json_data["asr_text"].strip()
        text_onehot = self.encode_text(text)
        text_len = len(text)
        label = self.class_label[image_path.split("/")[0]]
        return image_data, audio_data, audio_len, text_onehot, \
            text_len, label


class FlowerDataset(DatasetBase):
    def __init__(self, data_root, train=True):
        split = "train" if train else "test"
        self.json_data_all = json.load(open(os.path.join(data_root, "./{}.json".format(split)), "r"))
        image_embedding_path = self.json_data_all["image_feature_path"]
        print("load image embedding from:", image_embedding_path)
        self.image_embedding = pickle.load(open(image_embedding_path, "rb"))
        self.audio_folder = self.json_data_all["audio_base_path"]
        self.data = self.json_data_all['data']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        image_data, _ = self.get_rand(self.image_embedding[index])
        
        while True:
            try:
                text, rand_idx = self.get_rand(data['text'])
                text_onehot = self.encode_text(text)
                audio_data_path = os.path.join(self.audio_folder, data['wav'][rand_idx])
                audio_data, audio_len = load_one_audio_file(audio_data_path, {"target_length":2048})
                if audio_len >= 64:
                    break
                else:
                    continue
            except EOFError:
                print("data not found: {}".format(audio_data_path))
                continue
        text_len = len(text)
        label = int(data['class'])-1
        return image_data, audio_data, audio_len, text_onehot, text_len, label 


class BirdDataset(DatasetBase):
    def __init__(self, data_root, train=True):
        split = "train" if train else "test"
        self.json_data_all = json.load(open(os.path.join(data_root, "./{}.json".format(split)), 'r'))
        image_embedding_path = self.json_data_all['image_feature_path']
        print("load image embedding from:", image_embedding_path)
        self.image_embedding = pickle.load(open(image_embedding_path, 'rb'))
        self.audio_folder = self.json_data_all['audio_base_path']
        self.data = self.json_data_all['data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        image_data, _ = self.get_rand(self.image_embedding[index])
        
        while True:
            try:
                text, rand_idx = self.get_rand(data['text'])
                text_onehot = self.encode_text(text)
                audio_data_path = os.path.join(self.audio_folder, data['audio'][rand_idx])
                audio_data, audio_len = load_one_audio_file(audio_data_path, {"target_length":2048})
                if audio_len >= 64:
                    break
                else:
                    continue
            except EOFError:
                print("data not found: {}".format(audio_data_path))
                continue
        text_len = len(text)
        label = int(data['class'].split('.')[0])-1
        return image_data, audio_data, audio_len, text_onehot, text_len, label


def sort_torch_data(tuple_data:tuple, sort_idx:int)->tuple:
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(tuple_data[sort_idx], 0, True)
    rtn_data = tuple([d[sorted_cap_indices] for d in tuple_data])
    return rtn_data

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad.abs().mean()
    return hook

def batch_process(model, data, train_mode=True, **kwargs)->dict:
    image_feature, audio_data, n_frames, text_onehot, text_len, label = data
    image_feature, audio_data, label = image_feature.float().cuda(), audio_data.float().cuda(), label.long().cuda()
    if train_mode:
        # print(image_feature.shape, audio_data.shape)
        if kwargs.get('length_required', False):
            image_feature, audio_data, n_frames, text_onehot, text_len, label = \
                sort_torch_data((image_feature, audio_data, n_frames, text_onehot, text_len, label), sort_idx=2)
            _, sent_emb = model.forward(audio_data, n_frames//64)
            audio_feature = sent_emb
        else:
            audio_feature = model.forward(audio_data)
        # audio_feature = nn.Parameter(audio_feature)
        audio_feature.register_hook(save_grad('audio_feature'))
        optimizer = kwargs['optimizer']
        loss_func = kwargs['loss_func']
        loss = loss_func(audio_feature, image_feature, label)
        model.zero_grad()
        loss['loss'].backward()
        # grad_mean = audio_feature.grad.mean().item()
        optimizer.step()
        rtn = {
            "vars":{"loss":loss['loss'].item(), 'accu':loss['accu']}, 
            'count':{"loss":label.shape[0], 'accu':label.shape[0]}, 
            "output":"loss: {:.3f}({:.3f},{:.3f},{:.3f}), accu: {:.3f}, grad_mean:{:.7f}".format(
                loss['loss'].item(), loss['loss_jel'].item(), loss['loss_l1'].item(), 
                loss['loss_distill'].item(), loss['accu'], grads.get('audio_feature', 0)
                )
            }
    else:
        with torch.no_grad():
            if kwargs.get('length_required', False):
                image_feature, audio_data, n_frames, text_onehot, text_len, label = \
                    sort_torch_data((image_feature, audio_data, n_frames, text_onehot, text_len, label), sort_idx=2)
                _, sent_emb = model.forward(audio_data, n_frames//64)
                audio_feature = sent_emb
            else:
                audio_feature = model.forward(audio_data)
        loss_func = kwargs['loss_func']
        loss = loss_func(audio_feature.detach(), image_feature.detach(), label.detach())
        rtn = {
            "vars":{"loss":loss['loss'].item(), "accu":loss['accu']}, 
            'count':{"loss":label.shape[0], "accu":label.shape[0]},
            "output":"loss: {:.3f}({:.3f},{:.3f},{:.3f}), accu: {:.3f}".format(
                loss['loss'].item(), loss['loss_jel'].item(), loss['loss_l1'].item(), 
                loss['loss_distill'].item(), loss['accu']
                )
            }
    return rtn

@torch.no_grad()
class EvalClass(object):
    def __init__(self, recall_topk=50):
        self.recall_topk = 50
        self.best_accu = 0
        self.best_ap50 = 0
    @staticmethod
    def eval_class(query_feature_all, target_feature_all, label_all, topk):
        # query class center
        label_all_list = list(set(label_all))
        print("total eval class: {}".format(len(label_all_list)))
        label_all = np.array([label_all_list.index(label) for label in label_all])
        class_num = len(label_all_list)
        label_all_list = list(set(label_all))
        class_center_query = np.array([query_feature_all[label_all==label].mean(axis=0) for label in label_all_list])  # class_num x feature
    
        scores_all = np.matmul(target_feature_all, class_center_query.transpose()) # batch x class_num
        # scores_all = scores_all/np.expand_dims(abs(scores_all).sum(1), 1) # heart performance
        preds = scores_all.argmax(axis=1)  # batch
        accu_all = (preds==label_all).sum()/label_all.shape[0]
        topk_index_per_class = [np.argsort(scores_all[:,label], axis=0)[-topk:][::-1] for label in label_all_list]  # 50 x 50
        scores_per_class_50 = [scores_all[idx,:] for idx in topk_index_per_class]  #50 x 50
        labels_per_class_50 = [label_all[idx] for idx in topk_index_per_class] # 50 x 50
        # label_per_class_50 = [label[np.argsort(score)[-topk:][::-1]] for score,label in zip(scores_per_class, labels_per_class)]
        ap50 = np.array([(label==k).sum() for k, label  in enumerate(labels_per_class_50)]).sum()/np.array(labels_per_class_50).size
        # print(np.array(labels_per_class_50).size)
        return accu_all*100, ap50*100

    def __call__(self, model, val_dataloader, logger:logging.Logger, 
        writer:SummaryWriter, eval_iteration_idx, **kwargs):
        # extract feature
        audio_feature_all = []
        image_feature_all = []
        label_all = []
        for idx, data in enumerate(val_dataloader):
            image_feature, audio_data, n_frames, text_onehot, text_len, label = data
            image_feature, audio_data, label = image_feature.float().cuda(), audio_data.float().cuda(), label.long().cuda()
            with torch.no_grad():
                if kwargs.get('length_required', False):
                    # sort data
                    image_feature, audio_data, n_frames, text_onehot, text_len, label = \
                        sort_torch_data((image_feature, audio_data, n_frames, text_onehot, text_len, label), sort_idx=2)
                    _, sent_emb = model.forward(audio_data, n_frames//64)
                    audio_feature = sent_emb
                else:
                    audio_feature = model.forward(audio_data)
            audio_feature_all.append(audio_feature.detach().cpu().numpy())
            image_feature_all.append(image_feature.detach().cpu().numpy())
            label_all.append(label.detach().cpu().numpy())
        audio_feature_all, image_feature_all, label_all = np.concatenate(audio_feature_all),\
            np.concatenate(image_feature_all), np.concatenate(label_all)
        accu, ap = self.eval_class(audio_feature_all, image_feature_all, label_all, self.recall_topk)

        if logger is not None:
            logger.info("mode: test, accu: {:.3f}, ap50: {:.3f}".format(accu, ap))
        rtn = {"accu":accu, 'ap50':ap}
        if writer is not None:
            writer.add_scalars("test", rtn, global_step=eval_iteration_idx)
        return rtn
        


from trainer import save_checkpoint
class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
        self.best_ap50 = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        is_best = test_rtn.get('accu', 0) > self.best_accu
        self.best_accu = test_rtn.get('accu', 0) if is_best else self.best_accu
        self.best_ap50 = test_rtn.get('ap50', 0) if is_best else self.best_ap50
        model_filename = "epoch_{}.pth".format(epoch_idx)
        save_checkpoint(model, os.path.join(output_dir, model_filename), 
            meta={'epoch':epoch_idx})
        os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "latest.pth"))
            )
        if is_best:
            os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "best.pth"))
            )

        if logger is not None:
            logger.info("EvalHook: best accu: {:.3f}, best ap50: {:.3f}, is_best: {}".format(self.best_accu, self.best_ap50, is_best))

class LossFunc(object):
    def __init__(self, loss_diff, loss_same, jel_flag=True,
        l1_flag=True, lambda_l1=1, distill=True, distill_T=2, lambda_distill=1):
        if jel_flag:
            self.JEL = JointEmbeddingLossLayer(loss_diff_coeff=loss_diff, loss_same_coeff=loss_same)
        else:
            self.JEL = None
        if l1_flag:
            self.L1 = torch.nn.L1Loss()
        else:
            self.L1 = None
        self.lambda_l1 = lambda_l1
        if distill:
            self.distill = self.get_distill_loss
            self.distill_T = distill_T
        else:
            self.distill = None
        self.lambda_distill = lambda_distill

    def get_distill_loss(self, source, target):
        source, target = source, target.div(self.distill_T) # soft the target
        source, target = F.log_softmax(source, dim=1), F.softmax(target, dim=1)
        # loss = -(source*torch.log(target)).sum(dim=1).mean() *2*self.distill_T*self.distill_T
        loss = F.kl_div(source, target)
        return loss


    def __call__(self, source:torch.Tensor, target:torch.Tensor, label):
        if self.JEL is not None:
            jel = self.JEL(source, target, label)
        else:
            jel = {
                'loss':torch.cuda.FloatTensor([0]).requires_grad_(), 
                'grad':torch.cuda.FloatTensor([0]), 
                'accu':0
                }

        if self.L1 is not None:
            l1 = self.L1(source/torch.norm(source), target/torch.norm(target))
        else:
            l1 = torch.cuda.FloatTensor([0]).requires_grad_()
        if self.distill is not None:
            distill = self.distill(source, target) # self.compute_mmd(source, target) #
        else:
            distill = torch.cuda.FloatTensor([0]).requires_grad_()
        jel.update(
            {
                'loss':jel['loss']+l1*self.lambda_l1+distill*self.lambda_distill,
                'loss_jel':jel['loss'],
                'loss_l1':l1,
                'loss_distill':distill
            }
        )
        return jel

import random
def eval_audio_feature(audio_feature_path, image_feature_path, filename_path, topk=50):
    def load_pickle(filename):
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
        return data
    audio_features, image_features, filenames = load_pickle(audio_feature_path), load_pickle(image_feature_path), load_pickle(filename_path)
    data_len = len(filenames)
    audio_rand_idx = [random.randint(0,9) for _ in range(data_len)]
    image_rand_idx = [random.randint(0,9) for _ in range(data_len)]
    audio_features = [feature[idx,:] for feature, idx in zip(audio_features, audio_rand_idx)]
    image_features = [feature[idx,:] for feature, idx in zip(image_features, image_rand_idx)]

    labels = [int(filename.split('.')[0]) for filename in filenames]
    audio_features, image_features = np.stack(audio_features), np.stack(image_features)
    eval_obj = EvalClass()
    result = eval_obj.eval_class(audio_features, image_features, labels, topk=50)
    print(result)
    return result



def get_parser():
    parser = argparse.ArgumentParser("baseline")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="./data/birds/CUB_200_2011_audio/audio/0")
    parser.add_argument("--output_dir", type=str, default="./output/Audio_to_Image/log/baseline")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--epoch", type=int, default=100, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument('--loss_diff', type=float, default=1, help=' param for loss, default 1')
    parser.add_argument('--loss_same', type=float, default=1, help=' param for loss, default 1')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=30, help="")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.2, help="")
    parser.add_argument("--dataset", choices=["birds", "flowers", "places"], default="birds", help="")
    parser.add_argument("--jel_flag", action='store_true', default=False, help="")
    parser.add_argument("--l1_flag", action='store_true', default=False, help="")
    parser.add_argument("--distill_flag", action='store_true', default=False, help="")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="")
    parser.add_argument("--lambda_distill", type=float, default=1.0, help="")
    parser.add_argument("--distill_T", type=float, default=2.0, help="")
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    parser.add_argument("--bidirectional", action='store_true', default=False, help="")
    parser.add_argument("--rnn_layers", type=int, default=1, help="")

    args = parser.parse_args()
    args.project_root = os.getcwd()
    return args

from collections import OrderedDict
def load_state_dict(path:str, model:nn.Module):
    state_dict = torch.load(path)
    # create new OrderedDict that does not contain `module.`
    if 'module' in state_dict.keys():
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    # load params
    model.load_state_dict(new_state_dict)
    return model


def main(args):
    # init for distributed training
    print("rank:{}, init dataset".format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")  #, rank=0, world_size=1, init_method="tcp://127.0.0.1:29500")
    if args.dataset == "birds":
        data_root = "./data/birds"
        train_dataset = BirdDataset(data_root=data_root, train=True)
        val_dataset = BirdDataset(data_root=data_root, train=False)
    elif args.dataset == "places":
        data_root = "./data/Places_subset"
        train_dataset = PlaceSubSet(data_root=data_root, train=True)
        val_dataset = PlaceSubSet(data_root=data_root, train=False)
    elif args.dataset == "flowers":
        data_root = "./data/flowers"
        train_dataset = FlowerDataset(data_root=data_root, train=True)
        val_dataset = FlowerDataset(data_root=data_root, train=False)
    else:
        raise NotImplementedError
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    # define model
    print("rank:{}, define model".format(args.local_rank))
    model = CNNRNN(40, embedding_dim=1024, drop_prob=args.dropout, nhidden=1024, nsent=1024, bidirectional=args.bidirectional, rnn_layers=args.rnn_layers)
    batch_param = {'length_required':True, 'sorted_required':True}
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)
    # loss
    loss_func = LossFunc(args.loss_diff, args.loss_same, args.jel_flag, args.l1_flag, args.lambda_l1, args.distill_flag, args.distill_T, args.lambda_distill)
    eval_hook = EvalHook()
    test_func = EvalClass()
    print("rank:{}, define trainer".format(args.local_rank))
    trainer = ClassifierTrainer(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, 
        loss_func=loss_func, batch_process=batch_process, output_dir=args.output_dir, local_rank=args.local_rank,
        print_every=args.print_every, eval_every=args.eval_every, lr_scheduler=lr_scheduler,
        eval_hook=eval_hook, test_func=test_func, batch_param=batch_param, resume_from=args.resume
        )
    trainer.run(args.epoch)


if __name__=="__main__":
    args = get_parser()
    print(args)
    main(args)