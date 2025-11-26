from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn
from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        pass

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, model_s_base, model_t_base, model_s, model_t, optimizer, opt,
                simam, mse_loss, criterion_cls, criterion_kd,
                criterion_masker, criterion_spherical):
    """One epoch distillation"""
    model_s_base.train()
    model_t_base.eval() 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    attn_losses = AverageMeter()
    mask_losses = AverageMeter()
    sph_losses = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)


        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s_base(input, is_feat=True, preact=preact)
        _,_,proj_x_s, proj_s = model_s(input)
        with torch.no_grad():
            feat_t, logit_t = model_t_base(input, is_feat=True, preact=preact)
            _,_,proj_x_t, proj_t = model_t(input)
            feat_t = [f.detach() for f in feat_t]
        
        # ===================Loss=====================
        adapter_dynamic = None
        if simam is not None and mse_loss is not None:
            feat_s_attn, attn_s = simam(feat_s[-2])
            feat_t_attn, attn_t = simam(feat_t[-2])

            if feat_s_attn.shape[1] != feat_t_attn.shape[1]:
                if adapter_dynamic is None:
                    adapter_dynamic = nn.Conv2d(
                        in_channels=feat_s_attn.shape[1],
                        out_channels=feat_t_attn.shape[1],
                        kernel_size=1
                    ).to(feat_s_attn.device)
                feat_s_attn = adapter_dynamic(feat_s_attn)

            if feat_s_attn.shape[2:] != feat_t_attn.shape[2:]:
                feat_s_attn = F.interpolate(feat_s_attn, size=feat_t_attn.shape[2:], mode="bilinear", align_corners=False)

            loss_attn = mse_loss(feat_s_attn, feat_t_attn)
        else:
            loss_attn = 0.0

        loss_mask = criterion_masker(feat_s[-1], feat_t[-1], target)
        loss_spherical = criterion_spherical(proj_x_s, proj_x_t)
        loss_cls = criterion_cls(logit_s, target)

        if opt.distill == 'kd':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            
        total_loss = opt.cls_beta * loss_cls + opt.attn_beta * loss_attn + opt.mask_beta * loss_mask + opt.sph_beta * loss_spherical + opt.kd_beta * loss_kd
    
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(total_loss.item(), input.size(0))
        attn_losses.update(loss_attn.item(), input.size(0))
        mask_losses.update(loss_mask.item(), input.size(0))
        sph_losses.update(loss_spherical.item(), input.size(0))
        cls_losses.update(loss_cls.item(), input.size(0))
        kd_losses.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Attn {attn_loss.val:.4f}\t'
            'Mask {mask_loss.val:.4f}\t'
            'Sph {sph_loss.val:.4f}\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            .format(
                epoch, idx, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, attn_loss=attn_losses,
                mask_loss=mask_losses, sph_loss=sph_losses,
                top1=top1, top5=top5))

    return top1.avg, losses.avg, mask_losses.avg, attn_losses.avg, sph_losses.avg, cls_losses.avg, kd_losses.avg

import torch
import torch
import torch.nn.functional as F



def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    end = time.time()
    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
