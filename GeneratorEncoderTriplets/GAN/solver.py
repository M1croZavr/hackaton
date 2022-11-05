"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model
from checkpoint import CheckpointIO
from data_loader import InputFetcher
import utils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Solver(nn.Module):

    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        self.nets, self.nets_ema = build_model(args)
        # Below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay
                    )

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=False, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=False, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)
                ]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=False, **self.nets_ema)]

        self.to(DEVICE)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print(f'Initializing model {name} by He...')
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, 'train')

        # Resume from checkpoint if specified
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        for i in tqdm(range(args.resume_iter, args.total_iters)):
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, y_trg = inputs.x_ref, inputs.y_ref
            masks = None

            # Discriminator optimization
            d_loss, d_losses = compute_d_loss(
                nets, args, x_real, y_org, x_ref, y_trg, masks=masks
                )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # Generator optimization
            g_loss, g_losses = compute_g_loss(
                nets, args, x_real, y_org, x_ref, y_trg, masks=masks
                )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # Compute moving average models parameters (interpolate)
            moving_average(nets.generator, nets_ema.generator, beta=0.999)

            # Log training info
            if i % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses, g_losses],
                                        ['D/loss', 'G/loss']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                # all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # Save models state dictionaries
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)


def compute_d_loss(nets, args, x_real, y_org, x_ref, y_trg, masks=None):
    """Discriminator losses"""
    # with real images
    x_ref.requires_grad_()
    out = nets.discriminator(x_ref, y_trg)
    loss_real = adv_loss2(out, 1)
    loss_reg = r1_reg(out, x_ref)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss2(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())
    # with torch.no_grad():
    #     s_trg = nets.style_encoder(x_ref, y_trg)
    #     x_fake = nets.generator(x_real, s_trg, masks=masks)

    # x_ref.requires_grad_()
    # disc_real = nets.discriminator(x_ref, y_trg)
    # disc_fake = nets.discriminator(x_fake.detach(), y_trg)
    # disc_loss = adv_loss(disc_real, disc_fake)
    # reg_loss = r1_reg(disc_real, x_ref)

    # loss = disc_loss + args.lambda_reg * reg_loss
    # return loss, Munch(discriminator_loss=disc_loss.item(),
    #                    reg=reg_loss.item())


def compute_g_loss(nets, args, x_real, y_org, x_ref, y_trg, masks=None):
    """Generator losses"""
    s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss2(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       cyc=loss_cyc.item())
    # s_trg = nets.style_encoder(x_ref, y_trg)
    # x_fake = nets.generator(x_real, s_trg, masks=masks)
    # disc_real = nets.discriminator(x_ref, y_trg)
    # disc_fake = nets.discriminator(x_fake, y_trg)
    # loss_adv = adv_loss(disc_fake, disc_real)
    # # style reconstruction loss
    # s_pred = nets.style_encoder(x_fake, y_trg)
    # loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    # # cycle-consistency loss
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    # s_org = nets.style_encoder(x_real, y_org)
    # x_rec = nets.generator(x_fake, s_org, masks=masks)
    # loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # loss = loss_adv + args.lambda_sty * loss_sty + args.lambda_cyc * loss_cyc
    # return loss, Munch(adv=loss_adv.item(),
    #                    sty=loss_sty.item(),
    #                    cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    """
    Makes linear interpolation of parameters between two models, adjust parameters tensors by moving average
    param.data = param.data + beta * (param_test.data - param.data)
    """
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(real, fake):
    """Hinge adversarial loss on discriminator outputs"""
    zeros_tensor = torch.zeros(len(real)).view(-1, 1)
    zeros_tensor = zeros_tensor.to(fake.device)
    real = real.view(-1, 1)
    fake = fake.view(-1, 1)
    loss = torch.mean(torch.max(zeros_tensor, 1 + fake - real))
    return loss

def adv_loss2(logits, target):
    """Cross entropy discriminator loss"""
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
    