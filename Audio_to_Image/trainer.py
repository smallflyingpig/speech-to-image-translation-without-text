import time 
import logging 
import os 
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F 
from torch.utils.data  import Dataset, DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os.path as osp
from copy import deepcopy



def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    
    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict)
    else:
        model.load_state_dict(state_dict, strict)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.
    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))

    os.makedirs(osp.dirname(filename), exist_ok=True)
    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    torch.save(checkpoint, filename)

class RunningAverage(object):
    def __init__(self):
        self.val_hist = {}
        self.n_hist = {}

    def update(self, data:dict, count:dict):
        assert(isinstance(data, dict))
        assert(isinstance(count, dict))
        for key, value in data.items():
            self.val_hist[key] = self.val_hist.get(key, [])+[value]
            self.n_hist[key] = self.n_hist.get(key, [])+[count[key]]

    def clear(self):
        self.val_hist = {}
        self.n_hist = {}

    def average(self):
        avg = {}
        for key, value in self.val_hist.items():
            n = np.array(self.n_hist[key])
            v = np.array(value)
            avg[key] = (n*v).sum()/float(n.sum())
        return avg

class IncrementalAverage(object):
    def __init__(self, gamma=0.1):
        self.data = {}
        self.gamma = gamma

    def update(self, data:dict):
        assert(isinstance(data, dict))
        for key, value in data.items():
            value_base = self.data.get(key, value)
            self.data[key] = value_base + value*self.gamma

    def get_value(self)->dict:
        return self.data



from abc import ABCMeta, abstractmethod
class BaseTrainer(object):
    __metaclass__=ABCMeta
    def __init__(self):
        pass
    @abstractmethod
    def train_once(self):
        pass
    @abstractmethod
    def eval_once(self):
        pass
    @abstractmethod
    def run(self):
        pass
    @property
    def timestamp(self):
        return time.strftime('%Y%m%d_%H%M%S', time.localtime())

    def get_dist_info(self): # copy from mmcv
        if torch.__version__ < '1.0':
            initialized = dist._initialized
        else:
            initialized = dist.is_initialized()
        if initialized:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

class ClassifierTrainer(BaseTrainer):
    def __init__(self, model:nn.Module, train_dataloader:DataLoader, 
        optimizer, loss_func, batch_process, output_dir:str, local_rank:int, 
        val_dataloader:DataLoader=None, logger:logging.Logger=None, 
        writer:SummaryWriter=None, lr_scheduler=None, test_func=None,
        eval_every=5, print_every=50, resume_from=None, write_var_every=100, write_img_every=500,
        eval_hook=None, no_dist=False, batch_param={}):
        super(ClassifierTrainer, self).__init__()
        self.model = model
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
        self.optimizer, self.loss_func = optimizer, loss_func
        self.batch_process = batch_process
        self.test_func = test_func
        self.output_dir = output_dir
        self.local_rank = local_rank
        self.lr_scheduler, self.eval_every = lr_scheduler, eval_every
        self.print_every = print_every
        self.rank, self.world_size = self.get_dist_info()
        self.train_iteration = len(train_dataloader)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if logger is None:
            self.logger = self.init_logger(output_dir, level=logging.INFO if self.local_rank==0 else logging.WARNING)
        else:
            self.logger = logger
        self.writer = writer if local_rank == 0 else None
        self.write_var_every = write_var_every
        self.write_img_every = write_img_every
        self.start_epoch = 0
        self.iteration_idx = int(0)
        self.eval_iteration_idx = int(0)
        self.eval_hook = eval_hook
        self.no_dist = no_dist
        self.batch_param = batch_param
        if resume_from is not None and len(resume_from)>0:
            self.model, self.optimizer, self.start_epoch = self.resume_model(self.model, self.optimizer, resume_from, self.logger)
            self.iteration_idx = len(self.train_dataloader)*self.start_epoch
            self.eval_iteration_idx = len(self.val_dataloader)*self.start_epoch
    @staticmethod
    def resume_model(model, optimizer, path, logger=None):
        checkpoint = load_checkpoint(model, path)
        if logger is not None:
            logger.info("resume model from:{}".format(path))
        start_epoch = 0
        if 'epoch' in checkpoint.get('meta', {}).keys():
            start_epoch = checkpoint['meta']['epoch']
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, start_epoch

    def init_logger(self, log_dir=None, level=logging.INFO):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger(self.timestamp)
        logger.setLevel(level)
        if log_dir and self.rank == 0:
            log_file = os.path.join(log_dir, '{}.log'.format(self.timestamp))
            filehander = logging.FileHandler(log_file)
            logger.addHandler(filehander)
        return logger

    def process_batch_rtn(self, rtn, iter_idx, iter_all, epoch_idx, epoch, running_averager=None,  train_mode=True):
        mode = 'train' if train_mode else 'eval '
        mode_iter = self.iteration_idx if train_mode else self.eval_iteration_idx
        if 'output' in rtn.keys():
            if mode_iter % self.print_every == 0:
                self.logger.info("epoch: {:0>4d}/{:0>4d} [{:0>5d}/{:d}], mode:{}, {}".format(
                    epoch_idx, epoch, iter_idx, iter_all, mode, rtn['output']
                    ))
        if 'vars' in rtn.keys() and 'count' in rtn.keys():
            if running_averager is not None:
                running_averager.update(rtn['vars'], rtn['count'])
            if self.writer is not None and  mode_iter % self.write_var_every == 0:
                self.writer.add_scalars("loss", rtn['vars'], mode_iter)
        if self.writer is not None and 'image' in rtn.keys():
            if mode_iter % self.write_img_every == 0:
                for tag, tensor in rtn['image'].items():
                    self.writer.add_image(tag, torchvision.utils.make_grid(tensor, normalize=True), mode_iter)

    def train_once(self, epoch_idx, epoch):
        self.model.train()
        running_averager = RunningAverage()
        iter_all = len(self.train_dataloader)
        for iter_idx, data in enumerate(self.train_dataloader):
            rtn = self.batch_process(self.model, data, train_mode=True, 
                optimizer=self.optimizer, loss_func=self.loss_func, **self.batch_param)
            assert(isinstance(rtn, dict))
            self.process_batch_rtn(rtn, iter_idx, iter_all, epoch_idx, epoch, running_averager, train_mode=True)
            self.iteration_idx += 1
        self.logger.info("epoch: {:0>4d}, mode:train, lr: {:.4f}, {}".format(
                    epoch_idx, 
                    list(self.optimizer.param_groups)[0]['lr'], 
                    str(running_averager.average())[1:-2]
                    ))
            
    def eval_once(self, epoch_idx, epoch):
        self.model.eval()
        running_averager = RunningAverage()
        iter_all = len(self.val_dataloader)
        for iter_idx, data in enumerate(self.val_dataloader):
            rtn = self.batch_process(self.model, data, train_mode=False, 
                loss_func=self.loss_func, **self.batch_param)
            assert(isinstance(rtn, dict))
            self.process_batch_rtn(rtn, iter_idx, iter_all, epoch_idx, epoch, running_averager, train_mode=False)
            self.eval_iteration_idx += 1
        self.logger.info("epoch: {:0>4d} mode: eval, {}".format(
                        epoch_idx, str(running_averager.average())[1:-2]
                        ))
        return running_averager.average()

    def run(self, epoch, start=None):
        self.logger.info("start training...")
        self.logger.info("output dir: {}".format(self.output_dir))
        start = self.start_epoch if start is None else start
        for epoch_idx in range(start, epoch):
            
            self.train_once(epoch_idx, epoch)
            if (epoch_idx % self.eval_every == 0 or epoch_idx == epoch-1):
                eval_rtn = {}
                test_rtn = {}
                if self.val_dataloader is not None:
                    eval_rtn = self.eval_once(epoch_idx, epoch)
                if self.test_func is not None:
                    test_rtn = self.test_func(self.model, self.val_dataloader, self.logger, self.writer, epoch_idx, **self.batch_param)
                if self.local_rank == 0 and self.eval_hook is not None:
                    self.eval_hook(self.model, epoch_idx, self.output_dir, eval_rtn, test_rtn, self.logger, self.writer)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        time.sleep(1)
        self.logger.info("training end!")
    


class GANTrainer(ClassifierTrainer):
    def __init__(self, model:dict, train_dataloader:DataLoader, 
        optimizer:dict, loss_func, batch_process, output_dir:str, local_rank:int, 
        val_dataloader:DataLoader=None, logger:logging.Logger=None, 
        writer:SummaryWriter=None, lr_scheduler=None, test_func=None,
        eval_every=5, print_every=50, resume_from=None, write_var_every=100, write_img_every=500,
        eval_hook=None, no_dist=False, batch_param={}):
        super(GANTrainer, self).__init__(model=model, train_dataloader=train_dataloader, 
            optimizer=optimizer, loss_func=loss_func, batch_process=batch_process, output_dir=output_dir, local_rank=local_rank, 
            val_dataloader=val_dataloader, logger=logger, 
            writer=writer, lr_scheduler=lr_scheduler, test_func=test_func,
            eval_every=eval_every, print_every=print_every, resume_from=None, write_var_every=write_var_every, write_img_every=500,
            eval_hook=eval_hook, no_dist=no_dist)
        self.model_G = model['G']
        self.model_D = model['D']
        self.optimizer_G = self.optimizer.get('G', None)
        self.optimizer_D = self.optimizer.get('D', None)
        self.batch_pram = batch_param
        if resume_from is not None and len(resume_from)>0:
            filename_base, filename_ext = os.path.splitext(resume_from)
            resume_from_D = filename_base+'_Ds'+filename_ext
            self.model_D, self.optimizer_D, self.start_epoch = self.resume_mode(self.model_D, self.optimizer_D, resume_from_D, self.logger)
            resume_from_G = filename_base+'_G'+filename_ext
            self.model_G, self.optimizer_G, self.start_epoch = self.resume_mode(self.model_G, self.optimizer_G, resume_from_G, self.logger)
            self.iteration_idx = len(self.train_dataloader)*self.start_epoch
            self.eval_iteration_idx = len(self.val_dataloader)*self.start_epoch

        # print(self.model_G)
        # print(self.model_D)
        
    
    def train_once(self, epoch_idx, epoch):
        self.model_G.train()
        self.model_D.train()
        running_averager = RunningAverage()
        iter_all = len(self.train_dataloader)
        for iter_idx, data in enumerate(self.train_dataloader):
            rtn = self.batch_process(self.model_G, self.model_D, data, train_mode=True, 
                optimizer=self.optimizer, loss_func=self.loss_func, **self.batch_pram)
            assert(isinstance(rtn, dict))
            self.process_batch_rtn(rtn, iter_idx, iter_all, epoch_idx, epoch, running_averager, train_mode=True)
            self.iteration_idx += 1
        self.logger.info("epoch: {:0>4d}, mode:train, lr: (G: {:.4f}, D:{:.4f}), {}".format(
                    epoch_idx, 
                    list(self.optimizer_G.param_groups)[0]['lr'], list(self.optimizer_D.param_groups)[0]['lr'], 
                    str(running_averager.average())[1:-2]
                    ))

    @staticmethod
    def load_params(model:nn.Module, new_param):
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)
    @staticmethod
    def copy_G_params(model:nn.Module):
        flatten = deepcopy(list(p.data for p in model.parameters()))
        return flatten
    @staticmethod
    def running_average(target:list, update:list):
        for p, avg_p in zip(update, target):
            avg_p.mul_(0.999).add_(0.001, p.data)
        return target

    def run(self, epoch, start=None):
        self.logger.info("start training...")
        self.logger.info("output dir: {}".format(self.output_dir))
        avg_param_G = self.copy_G_params(self.model_G)
        start = self.start_epoch if start is None else start
        for epoch_idx in range(start, epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.train_once(epoch_idx, epoch)
            avg_param_G = self.running_average(avg_param_G, self.model_G.parameters())
            if (epoch_idx % self.eval_every == 0 or epoch_idx == epoch-1):
                test_rtn = {}
                param_backup = self.copy_G_params(self.model_G)
                self.load_params(self.model_G, avg_param_G)
                if self.test_func is not None:
                    test_rtn = self.test_func(self.model_G, self.val_dataloader, self.logger, self.writer, epoch_idx)
                if self.local_rank == 0 and self.eval_hook is not None:
                    self.eval_hook(self.model_G, self.model_D, epoch_idx, self.output_dir, 
                        eval_rtn=None, test_rtn=test_rtn, logger=self.logger, writer=self.writer)
                self.load_params(self.model_G, param_backup)
            
        time.sleep(1)
        self.logger.info("training end!")
