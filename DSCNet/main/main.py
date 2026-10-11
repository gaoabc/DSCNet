import argparse
import logging
import os
import random
import shutil
import sys
from typing import Iterable
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from networks.unet_model import UNet
from preprocessing.dataloader import EECSegmentation, EGCSegmentation, KSSegmentation, ISICSegmentation, \
    CVC_ClinicDBSegmentation
import preprocessing.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt
from utils.contrastive import compute_class_centroids, confidence_aware_contrastive_loss
from utils.IB import HSIC_IB_Loss
from torch.optim.lr_scheduler import LambdaLR
import math

def get_current_consistency_weight(epoch):

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def cycle(iterable: Iterable):

    while True:
        for x in iterable:
            yield x


def get_SGD(net, name='SGD', lr=0.03, momentum=0.9, \
            weight_decay=e-4, nesterov=True, bn_wd_skip=True):
    optim = getattr(torch.optim, name)

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': e-4}]

    optimizer = optim(per_param_args, lr=lr,
                      momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def extract_amp_spectrum(img_np):
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)
    return amp_np


def low_freq_mutate_np(amp_src, amp_trg, L=0.1, degree=1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.uniform(0, degree)

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * (1 - ratio) + a_trg[:, h1:h2, w1:w2] * ratio
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1, degree=1):

    src_img = src_img
    src_img_np = src_img
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L, degree=degree)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    return src_in_trg  # .transpose(1, 2, 0)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def cycle(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def get_SGD(net, name='SGD', lr=0.03, momentum=0.9, weight_decay=e-4, nesterov=True, bn_wd_skip=True):
    optim = getattr(torch.optim, name)
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': e-4}]
    optimizer = optim(per_param_args, lr=lr,
                      momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_cycles=7. / 16., num_warmup_steps=0,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def select_dataset(args):
    if args.dataset == 'CVC-ClinicDB':
        dataset = CVC_ClinicDBSegmentation
    elif args.dataset == 'EEC':
        dataset = EECSegmentation
    elif args.dataset == 'EGC':
        dataset = EGCSegmentation
    elif args.dataset == 'ISIC':
        dataset = ISICSegmentation
    elif args.dataset == 'KS':
        dataset = KSSegmentation
    return dataset
part=['base']
n_part = len(part)
dice_calcu = metrics.dice_coeff


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_metrics = []
    domain_num = len(test_dataloader)

    if args.dataset in ['CVC-ClinicDB', 'EGC', 'EEC','Polyp', 'ISIC']:
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False

    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        for batch_num, sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()

            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset in ['CVC-ClinicDB', 'EGC', 'EEC', 'Polyp', 'ISIC']:
                mask = mask.eq(255).long()
            output = model(data)
            loss_seg = ce_loss(output, mask).mean() + \
                       dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)


            if args.dataset in ['CVC-ClinicDB', 'EGC', 'EEC', 'Polyp', 'ISIC']:
                dice = dice_calcu[args.dataset](np.asarray(torch.max(torch.softmax(output.cpu(), dim=1), dim=1)[1]),
                                                mask.clone().cpu())

            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]

        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part - 1:
                text += ','
        logging.info(text)

    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part - 1:
            text += ','
    logging.info(text)
    return val_dice


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    dataset = select_dataset(args)
    test_dataset = []
    test_dataloader = []
    domain_num = args.domain_num
    weak = transforms.Compose(
        [tr.RandomScaleCrop(256), tr.RandomScaleRotate(), tr.RandomHorizontalFlip(), tr.RGBToHSL(),
         tr.elastic_transform()])
    strong = transforms.Compose([tr.RGBToHSL1(), tr.Brightness(0.5, 1.5), tr.Contrast(0.5, 1.5),
                                 tr.GaussianBlur(kernel_size=25, num_channels=3)])

    lb_dataset = dataset(base_dir=train_data_path, phase='train', selected_idxs=range(300), weak_transform=weak,
                         strong_tranform=strong)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', selected_idxs=range(300, 600), weak_transform=weak,
                          strong_tranform=strong)
    lb_dataloader = DataLoader(lb_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    ulb_dataloader = DataLoader(ulb_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    for i in range(1, domain_num + 1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i],)
        test_dataset.append(cur_dataset)
    lb_dataloader = cycle(
        DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=16, pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(
        DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=16, pin_memory=True,
                   drop_last=True))
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size=args.test_bs, shuffle=False, num_workers=16,
                                    pin_memory=True)
        test_dataloader.append(cur_dataloader)

    model = UNet(n_channels=3, n_classes=2).cuda()
    ema_model = UNet(n_channels=3, n_classes=2).cuda()

    optimizer = get_SGD(model, lr=base_lr, momentum=0.9, weight_decay=e-4)

    ce_loss = CrossEntropyLoss(reduction='none')
    dice_loss = losses.DiceLossWithMask(2)
    hsic = HSIC_IB_Loss()

    max_epoch = args.max_iterations // args.num_eval_iter
    for epoch_num in range(max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num + 1}')
        for i_batch in range(1, args.num_eval_iter + 1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'].cuda(), lb_sample['label'].cuda()
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'].cuda(), ulb_sample['strong_aug'].cuda(), ulb_sample[
                'label'].cuda()
            s_pred, ib_total = model(lb_x_w)
            sup_loss = ce_loss(s_pred, lb_y) + dice_loss(s_pred, lb_y.unsqueeze(1))
            unsup_loss = (ce_loss(s_pred, ulb_y) * (ulb_y > 0.9).float()).mean()
            centroids = compute_class_centroids(s_pred, lb_y, num_classes=2)
            ctr_loss = confidence_aware_contrastive_loss(s_pred, lb_y, centroids)
            ib = hsic(lb_x_w,ib_total) - hsic(lb_y,ib_total)
            loss = sup_loss + unsup_loss +  ctr_loss +  ib
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p_bar.set_description(f'iteration {i_batch}: loss={loss.item():.3f}')
            writer.add_scalar('train/loss', loss.item(), epoch_num * args.num_eval_iter + i_batch)

        logging.info(f'Epoch {epoch_num + 1}: Testing EMA model...')
        test(args, ema_model, test_dataloader, epoch_num, writer, ema=True)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EEC',
                        choices=['EEC', 'EGC', 'KS', 'ISIC', 'CVC-ClinicDB'])
    parser.add_argument("--save_name", type=str, default="test")
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument('--num_eval_iter', type=int, default=100)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.03,)
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--gpu", type=str, default='3')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_path', type=str, default='/mnt/data2/home/DSCNet/models/iter_best.pth')
    parser.add_argument("--threshold", type=float, default=0.90, )
    parser.add_argument('--amp', type=int, default=1,)
    parser.add_argument("--label_bs", type=int, default=8,)
    parser.add_argument("--unlabel_bs", type=int, default=8,)
    parser.add_argument("--test_bs", type=int, default=16,)
    parser.add_argument('--domain_num', type=int, default=1)
    parser.add_argument('--lb_domain', type=int, default=1)
    parser.add_argument('--lb_num', type=int, default=269)
    parser.add_argument("--consistency_type", type=str, default="mse")
    parser.add_argument("--consistency", type=float, default=1.0)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--cutmix_prob", default=1.0, type=float)
    parser.add_argument("--LB", default=0.01, type=float)
    args = parser.parse_args()
    snapshot_path = "/mnt/data2/home/DSCNet/model/" + args.dataset + "/" + args.save_name + "/"
    if args.dataset == 'EEC':
        train_data_path = "/mnt/data2/home/DSCNet/data/EEC"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))
    train(args, snapshot_path)
