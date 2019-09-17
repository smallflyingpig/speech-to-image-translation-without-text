from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import tqdm
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p

from scipy import linalg

from tensorboardX import SummaryWriter

from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3

# add my finetune inception v3
import sys
import os
sys.path.append(os.getcwd())
from Audio_to_Image.Inception_v3 import Inception3_CUB



# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width
    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)
    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add(logvar)
    KLD = torch.mean(KLD_element).mul(-0.5)
    return KLD

def get_inception_loss(activate_r, activate_g):
    pairwise = activate_g-activate_r
    loss = torch.mean((pairwise**2))
    return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def compute_frethet_distance(predictions_g, predictions_r, eps=1e-6):
    '''
    reference:https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    '''
    mu_g, sigma_g = np.mean(predictions_g, axis=0), np.cov(predictions_g, rowvar=False)
    mu_r, sigma_r = np.mean(predictions_r, axis=0), np.cov(predictions_r, rowvar=False)

    mu1 = np.atleast_1d(mu_g)
    mu2 = np.atleast_1d(mu_r)

    sigma1 = np.atleast_2d(sigma_g)
    sigma2 = np.atleast_2d(sigma_r)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    data=[
        {'mu':mu1, 'sigma':sigma1},
        {'mu':mu1, 'sigma':sigma2}
    ]

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean), data



def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus:list, distributed:bool):
    netG = G_NET()
    netG.apply(weights_init)
    if distributed:
        netG = netG.cuda()
        netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=gpus, output_device=gpus[0], broadcast_buffers=True)
    else:
        if cfg.CUDA:
            netG = netG.cuda()
        netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    if cfg.TREE.BRANCH_NUM > 3:
        netsD.append(D_NET512())
    if cfg.TREE.BRANCH_NUM > 4:
        netsD.append(D_NET1024())
    # TODO: if cfg.TREE.BRANCH_NUM > 5:
    # netsD_module = nn.ModuleList(netsD)
    # netsD_module.apply(weights_init)
    # netsD_module = torch.nn.parallel.DistributedDataParallel(netsD_module.cuda(), device_ids=gpus, output_device=gpus[0])
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        if distributed:
            netsD[i] = torch.nn.parallel.DistributedDataParallel(netsD[i].cuda(), device_ids=gpus, output_device=gpus[0], broadcast_buffers=True
                # , process_group=pg_Ds[i]
                )
        else:
            netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i), map_location=lambda storage, loc: storage)
            netsD[i].load_state_dict(state_dict)

    if cfg.INCEPTION_CUB:
        inception_model = Inception3_CUB(num_classes=200) #INCEPTION_V3()
        inception_model.load_state_dict(torch.load("./StackGAN-v2/model/inception-v3_best.pt", map_location=torch.device("cuda:{}".format(gpus[0]))))
    else:
        inception_model = INCEPTION_V3()

    
    if not distributed:
        if cfg.CUDA:
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
            inception_model = inception_model.cuda()
        inception_model = torch.nn.DataParallel(inception_model, device_ids=gpus)
    else:
        inception_model = torch.nn.parallel.DistributedDataParallel(inception_model.cuda(), device_ids=gpus, output_device=gpus[0])
        pass
    # inception_model = inception_model.cpu() #to(torch.device("cuda:{}".format(gpus[0])))
    inception_model.eval()
    print("model device, G:{}, D:{}, incep:{}".format(netG.device_ids, netsD[0].device_ids, inception_model.device_ids))
    return netG, netsD, len(netsD), inception_model, count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    # real_img_set = vutils.make_grid(real_img).numpy()
    # real_img_set = np.transpose(real_img_set, (1, 2, 0))
    # real_img_set = real_img_set * 255
    # real_img_set = real_img_set.astype(np.uint8)
    # sup_real_img = summary.image('real_img', real_img_set)
    # summary_writer.add_summary(sup_real_img, count)
    summary_writer.add_image(tag="real_img", img_tensor=vutils.make_grid(real_img, normalize=True), global_step=count)


    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)

        summary_writer.add_image(tag="fake_{:d}_img".format(i), img_tensor=vutils.make_grid(fake_img, normalize=True), global_step=count)


def class_aware_loss(x_activates, class_labels):
    # x_activate: (batch, feature)
    # class_labels: (batch, 1)
    batch_size, feature_dim = x_activates.shape
    
    # print("x_active shape:", x_activates.shape)
    scores = torch.mm(x_activates, torch.transpose(x_activates, 0, 1))
    pair_matrix = torch.ByteTensor([a==b for a in class_labels for b in class_labels]).reshape((batch_size, batch_size))
    pair_matrix = pair_matrix - torch.diag(pair_matrix[torch.eye(batch_size).byte()]).to(pair_matrix.device)
    if pair_matrix.sum()>0:
        loss = torch.max(torch.Tensor([0]).to(scores.device), scores.mean()-scores[pair_matrix].mean()).div(feature_dim)
        return loss
    else:
        return torch.Tensor([0]).to(scores.device)


# ################# Text to image task############################ #



class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize, my_dataset_flag, local_rank=0, distributed=False):
        self.my_dataset_flag = my_dataset_flag
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)
        else:  #eval
            output_dir += "_eval"
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)
    
        # s_gpus = cfg.GPU_ID.split(',')
        # self.gpus = [int(ix) for ix in s_gpus]
        # self.num_gpus = len(self.gpus)
        # 
        # torch.cuda.set_device(self.gpus[0])
        # cudnn.benchmark = True
        self.gpus = [local_rank]

        self.batch_size = cfg.TRAIN.BATCH_SIZE  #  * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.distributed = distributed

    def prepare_data(self, data):
        if self.my_dataset_flag:
            imgs, w_imgs, t_embedding, _, class_labels = data["real_image"], data["wrong_image"], data["real_embedding"], data["text"]
        else:
            imgs, w_imgs, t_embedding, _, class_labels = data

        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = t_embedding.float().requires_grad_().cuda()
        else:
            vembedding = t_embedding.float().requires_grad_()
        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append((imgs[i]).cuda().requires_grad_())
                wrong_vimgs.append((w_imgs[i]).cuda().requires_grad_())
            else:
                real_vimgs.append((imgs[i]).requires_grad_())
                wrong_vimgs.append((w_imgs[i]).requires_grad_())
        return imgs, real_vimgs, wrong_vimgs, vembedding, class_labels

    def train_Dnet(self, idx, count):
        flag = count % cfg.TRAIN.LOG_INTERVAL
        batch_size = self.real_imgs[0].size(0)
        criterion, mu = self.criterion, self.mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        #
        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logits, real_x_active = netD(real_imgs, mu.detach())
        wrong_logits, wrong_x_active = netD(wrong_imgs, mu.detach())
        fake_logits, fake_x_active = netD(fake_imgs.detach(), mu.detach())
        #
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(fake_logits[1], fake_labels)
            #
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            #
            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward()
        # update parameters
        optD.step()
        # log
        if flag == 0:
            self.summary_writer.add_scalars(
                main_tag="D_loss",
                tag_scalar_dict={
                    'D_loss{:d}'.format(idx):errD
                },
                global_step=count
            )
            #self.summary_writer.add_scalar('D_loss{:d}'.format(idx), errD, count)
            
        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        errG_cal_total = 0
        flag = count % cfg.TRAIN.LOG_INTERVAL
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]
        for i in range(self.num_Ds):
            outputs, x_active = self.netsD[i](self.fake_imgs[i], mu)
            errG = criterion(outputs[0], real_labels)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS *\
                    criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            if cfg.TRAIN.COEFF.CAL_LOSS > 0:
                errG_cal = class_aware_loss(x_active, self.class_labels)
                errG_cal_total = errG_cal_total + errG_cal
            errG_total = errG_total + errG
            if flag == 0:
                self.summary_writer.add_scalars('G_loss', {
                    'G_loss{:d}'.format(i):errG
                }, count)
                #self.summary_writer.add_scalar('G_loss{:d}'.format(i), errG, count)

        # Compute color consistency losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
                if flag == 0:
                    self.summary_writer.add_scalar('G_like_mu2', like_mu2, count)
                    self.summary_writer.add_scalar('G_like_cov2', like_cov2, count)
                
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
                if flag == 0:
                    self.summary_writer.add_scalar('G_like_mu1', like_mu1, count)
                    self.summary_writer.add_scalar('G_like_cov1', like_cov1, count)

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        if flag == 0:
            self.summary_writer.add_scalars('G_loss', {
                'kl_loss':kl_loss,
                'cal_loss':errG_cal_total
            }, count)
        errG_total = errG_total + kl_loss + errG_cal_total
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus, self.distributed)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()

        self.real_labels = \
            (torch.FloatTensor(self.batch_size).fill_(1).requires_grad_(False))
        self.fake_labels = \
            (torch.FloatTensor(self.batch_size).fill_(0).requires_grad_(False))

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = (torch.FloatTensor(self.batch_size, nz).requires_grad_(False))
        fixed_noise = \
            (torch.FloatTensor(self.batch_size, nz).normal_(0, 1).requires_grad_())

        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions_g = []
        predictions_r = []
        activate_g = []
        activate_r = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            bar = tqdm.tqdm(self.data_loader) if self.gpus[0] == 0 else self.data_loader
            for step, data in enumerate(bar, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, \
                    self.txt_embedding, self.class_labels = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                noise_input = noise[:self.txt_embedding.shape[0]].requires_grad_(True)
                self.fake_imgs, self.mu, self.logvar = \
                    self.netG(noise_input, self.txt_embedding) # fix the bug for last batch
                # self.fake_imgs[0].mean().backward(retain_graph=True)
                # self.fake_imgs[1].mean().backward(retain_graph=True)
                # self.fake_imgs[2].mean().backward(retain_graph=True)
                
                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total = errD_total + errD

                # for inception score
                # with torch.autograd.no_grad():
                #     pred_g, pool3_g = self.inception_model(self.fake_imgs[-1].detach())
                #     pred_r, pool3_r = self.inception_model(self.real_imgs[-1].detach())
                
                with torch.autograd.no_grad():
                    pred_g, pool3_g = self.inception_model(self.fake_imgs[-1])
                    pred_r, pool3_r = self.inception_model(self.real_imgs[-1])
                    
                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                predictions_g.append(pred_g.detach().requires_grad_(False).cpu().data.numpy())
                predictions_r.append(pred_r.detach().requires_grad_(False).cpu().data.numpy())
                activate_g.append(pool3_g.detach().requires_grad_(False).cpu().data.numpy())
                activate_r.append(pool3_r.detach().requires_grad_(False).cpu().data.numpy())

                if count % 100 == 0:
                    self.summary_writer.add_scalars('D_loss', {'total':errD_total}, count)
                    self.summary_writer.add_scalars('G_loss', {'total':errG_total}, count)
                    
                if step == 0:
                    print('''[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f'''
                           % (epoch, self.max_epoch, step, self.num_batches,
                              errD_total.item(), errG_total.item()))
                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    fixed_noise_input = fixed_noise[:self.txt_embedding.shape[0]].requires_grad_(True)
                    with torch.autograd.no_grad():
                        self.fake_imgs, _, _ = \
                            self.netG(fixed_noise_input, self.txt_embedding)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                    count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions_g) >= 500:
                        print("len pred: {}".format(len(predictions_g)))
                        predictions_g = np.concatenate(predictions_g, 0)
                        predictions_r = np.concatenate(predictions_r, 0)
                        activate_g = np.concatenate(activate_g, 0)
                        activate_r = np.concatenate(activate_r, 0)

                        mean, std = compute_inception_score(predictions_g, 10)
                        # print('mean:', mean, 'std', std)
                        self.summary_writer.add_scalar('Inception_mean', mean, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions_g, 10)
                        self.summary_writer.add_scalar('NLPP_mean', mean_nlpp, count)
                        # FID
                        fid, _ = compute_frethet_distance(activate_g, activate_r)
                        self.summary_writer.add_scalar("FID", fid, count)
                        del predictions_g
                        del predictions_r
                        del activate_g
                        del activate_r
                        #
                        predictions_g = []
                        predictions_r = []
                        activate_g = []
                        activate_r = []
            bar.close()
            end_t = time.time()
            print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs'''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.cpu().item(), errG_total.cpu().item(),
                     kl_loss.cpu().item(), end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize, sample_idx=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                # print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d_%d.png' % (s_tmp, imsize, sentenceID, sample_idx)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    @torch.no_grad()
    def evaluate(self, split_dir):
        sample_num_per_image = 10
        if cfg.INCEPTION_CUB:
            self.inception_model = Inception3_CUB(num_classes=200).eval() #INCEPTION_V3().eval()
            self.inception_model.load_state_dict(torch.load("./StackGAN-v2/model/inception-v3_best.pt", map_location=lambda storage, loc: storage))
        else:
            self.inception_model = INCEPTION_V3().eval()

        if cfg.CUDA:
            self.inception_model = self.inception_model.cuda()
            # self.inception_model = torch.cuda.parallel.DistributedDataParallel(self.inception_model, device_ids=self.gpus, output_device=self.gpus[0])
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if split_dir == 'test':
                split_dir = 'valid'
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise_raw = (torch.FloatTensor(self.batch_size, nz).requires_grad_())
            if cfg.CUDA:
                netG.cuda()
                noise_raw = noise_raw.cuda()
            else:
                pass

            predictions_g = []
            predictions_r = []
            activate_g = []
            activate_r = []
            IS_mean_add = 0
            FID_add = 0
            NLPP_mean_add = 0
            metric_cnt = 0
            count = 0
            # switch to evaluate mode
            netG.eval()
            loader_bar = tqdm.tqdm(self.data_loader)
            for step, data in enumerate(loader_bar, 0):
                count += 1
                imgs, t_embeddings, filenames = data
                # print(t_embeddings.shape)
                if cfg.CUDA:
                    if isinstance(t_embeddings, list):
                        t_embeddings = [emb.requires_grad_(False).float().cuda() for emb in t_embeddings]
                    else:   
                        t_embeddings = t_embeddings.requires_grad_(False).float().cuda()
                    if isinstance(imgs, list):
                        imgs = [img.requires_grad_(False).float().cuda() for img in imgs]
                    else:
                        imgs = imgs.requires_grad_(False).float().cuda()
                else:
                    if isinstance(t_embeddings, list):
                        t_embeddings = [emb.requires_grad_(False).float() for emb in t_embeddings]
                    else:   
                        t_embeddings = t_embeddings.requires_grad_(False).float()
                    if isinstance(imgs, list):
                        imgs = [img.requires_grad_(False).float() for img in imgs]
                    else:
                        imgs = imgs.requires_grad_(False).float()
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise = noise_raw[:batch_size]
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)
                # trunc
                # noise[noise<-cfg.TEST.TRUNC] = -cfg.TEST.TRUNC
                # noise[noise>cfg.TEST.TRUNC] = cfg.TEST.TRUNC

                fake_img_list = []

                for i in range(embedding_dim):
                    for sample_idx in range(sample_num_per_image):
                        noise.data.normal_(0, 1)
                        with torch.autograd.no_grad():
                            fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
                            real_imgs = imgs
                        # pred_g, pool3_g = self.inception_model(fake_imgs[-1].detach())
                        # pred_r, pool3_r = self.inception_model(real_imgs[-1].detach())
                    # predictions_g.append(pred_g.data.cpu().numpy())
                    # predictions_r.append(pred_r.data.cpu().numpy())
                    # activate_g.append(pool3_g.data.cpu().numpy())
                    # activate_r.append(pool3_r.data.cpu().numpy())

                        if cfg.TEST.B_EXAMPLE:
                            # fake_img_list.append(fake_imgs[0].data.cpu())
                            # fake_img_list.append(fake_imgs[1].data.cpu())
                            fake_img_list.append(fake_imgs[2].data.cpu())
                        else:
                            self.save_singleimages(fake_imgs[-1], filenames,
                                                   save_dir, split_dir, i, 256, sample_idx)
                            # self.save_singleimages(fake_imgs[-2], filenames,
                            #                        save_dir, split_dir, i, 128)
                            # self.save_singleimages(fake_imgs[-3], filenames,
                            #                        save_dir, split_dir, i, 64)
                    # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)
            print("len pred: {}".format(len(predictions_g)))
            return [{'mu':0, 'sigma':0},{'mu':0, 'sigma':0}]  # to save time

            predictions_g = np.concatenate(predictions_g, 0)
            predictions_r = np.concatenate(predictions_r, 0)
            activate_g = np.concatenate(activate_g, 0)
            activate_r = np.concatenate(activate_r, 0)
            mean, std = compute_inception_score(predictions_g, 10)
            # print('mean:', mean, 'std', std)
            self.summary_writer.add_scalar('Inception_mean', mean, count)
            #
            mean_nlpp, std_nlpp = \
                negative_log_posterior_probability(predictions_g, 10)
            self.summary_writer.add_scalar('NLPP_mean', mean_nlpp, count)
            # FID
            fid, fid_data = compute_frethet_distance(activate_g, activate_r)
            self.summary_writer.add_scalar("FID", fid, count)
            IS_mean_add += mean
            FID_add += fid
            NLPP_mean_add += mean_nlpp
            metric_cnt += 1

            IS_mean, FID_mean, NLPP_mean = IS_mean_add/metric_cnt, FID_add/metric_cnt, NLPP_mean_add/metric_cnt
            print("total, IS mean:{}, FID mean:{}, NLPP mean:{}".format(IS_mean, FID_mean, NLPP_mean))

            return fid_data