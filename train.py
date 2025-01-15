#File: train.py

import time
from options.train_options import TrainOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from tqdm import tqdm
import wandb  # Added import
import torch  # Added import
import socket  # Added import for connectivity check
from models.CCLNet.Public.util.AverageMeter import AverageMeter
from metrics import (
    compute_psnr_batch, 
    compute_ssim_batch,
    compute_uiqm_batch,
    compute_uciqe_batch
)


def is_online():
    try:
        socket.create_connection(("8.8.8.8", 53), 2)
        return True
    except:
        return False
def validate_model(opt, model, epoch, visualizer):
    """
    A small validation loop that:
     1) Creates a val-dataset with opt.phase='val'.
     2) Iterates over the dataset
     3) Collects metrics (PSNR, SSIM, UIQM, UCIQE)
     4) Optionally collects a validation loss if references exist
     5) Logs results (WandB, visualizer)
    """
    # <<< NEW: Make a copy of `opt` to not overwrite training-phase.
    from copy import deepcopy
    val_opt = deepcopy(opt)
    val_opt.phase = 'val'  # or 'test' if you want to reuse your test set as "val"
    val_opt.isTrain = False
    val_opt.no_flip = True
    val_opt.load_size = 256  # typical for validation
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.batch_size = 1  # typical for validation

    val_dataset = create_dataset(val_opt)
    val_size = len(val_dataset)
    visualizer.print_msg(f"--- Start validation: #images={val_size} ---")

    if val_size == 0:
        visualizer.print_msg(f"No validation data found. Skipping validation.")
        return

    # switch to eval mode
    model.eval()

    # accumulators
    psnr_list, ssim_list = [], []
    uiqm_list, uciqe_list = [], []
    val_loss_meter = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            # 1) color-correct raw -> rgb_fcc
            # 2) forward through hrnet -> pred_hr
            model.forward()

            # If you have a ref in the val set, you can do the same logic as training
            # for computing a loss:
            if 'ref' in data.keys():
                model.compute_loss()
                if model.loss_all is not None:
                    val_loss_meter.update(model.loss_all.item())

            # 3) Convert pred_hr from [-1,1] or lab-scaling to [0,1] if needed
            #    Typically, 'pred_hr' is in [-1,1] from the final Tanh,
            #    so we do the same as `tensor2im()` but keep it as a tensor.
            print(f"pred_hr_range: {model.pred_hr.min()}, {model.pred_hr.max()}")
            pred_hr = (model.pred_hr + 1) / 2.0  # shape [B, C, H, W], in [0,1]
            print(f"pred_hr_range: {pred_hr.min()}, {pred_hr.max()}")
            # Possibly clamp:
            pred_hr = torch.clamp(pred_hr, 0.0, 1.0)
            print(f"pred_hr_range: {pred_hr.min()}, {pred_hr.max()}")

            if 'ref' in data.keys():
                # We also convert ref from LAB->RGB in training code:
                print(f" ref lab image range: {data['ref'].min()}, {data['ref'].max()}")
                ref_rgb = model.lab2rgb.labn12p1_to_rgbn12p1(data['ref'].to(model.device))
                print(f" ref rgb image range: {ref_rgb.min()}, {ref_rgb.max()}")
                ref_rgb = (ref_rgb + 1)/2.0
                print(f" ref rgb image range: {ref_rgb.min()}, {ref_rgb.max()}")
                ref_rgb = torch.clamp(ref_rgb, 0.0, 1.0)
                print(f" ref rgb image range: {ref_rgb.min()}, {ref_rgb.max()}")

                # Compute PSNR/SSIM if we have a reference
                psnr_vals = compute_psnr_batch(pred_hr, ref_rgb, data_range=1.0, reduction='none')
                ssim_vals = compute_ssim_batch(pred_hr, ref_rgb, data_range=1.0, reduction='none')
                # average across batch dimension
                # but typically batch=1 in val
                psnr_list.extend(psnr_vals)
                ssim_list.extend(ssim_vals)

            # Compute UIQM, UCIQE (no ref needed, so always compute)
            uiqm_vals = compute_uiqm_batch(pred_hr, reduction='none')   # list
            uciqe_vals = compute_uciqe_batch(pred_hr, reduction='none') # list
            uiqm_list.extend(uiqm_vals)
            uciqe_list.extend(uciqe_vals)

    # done with val dataset
    mean_psnr  = float(np.mean(psnr_list))  if psnr_list else 0.0
    mean_ssim  = float(np.mean(ssim_list))  if ssim_list else 0.0
    mean_uiqm  = float(np.mean(uiqm_list))  if uiqm_list else 0.0
    mean_uciqe = float(np.mean(uciqe_list)) if uciqe_list else 0.0

    val_loss = val_loss_meter.avg if val_loss_meter.count>0 else 0.0

    # log:
    visualizer.print_msg(
      f"[Val] epoch={epoch}, loss={val_loss:.4f} "
      f"PSNR={mean_psnr:.3f}, SSIM={mean_ssim:.3f}, "
      f"UIQM={mean_uiqm:.3f}, UCIQE={mean_uciqe:.3f}"
    )

    # wandb log:
    wandb.log({
        "val_loss":  val_loss,
        "val_psnr":  mean_psnr,
        "val_ssim":  mean_ssim,
        "val_uiqm":  mean_uiqm,
        "val_uciqe": mean_uciqe,
        "epoch": epoch
    })

    # Switch back to train mode:
    model.train()


def train_model(opt, model):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    best_loss = 9999.0
    best_epoch = 0

    # Initialize wandb with API key
    wandb.login(key='8d9ec67ac85ce634d875b480fed3604bfb9cb595')  # Added wandb login

    try:
        if is_online():
            print("Internet detected, using wandb online mode.")
            wandb.init(
                project='CCHRNET',
                config=opt,
                name=opt.name,  # Use the --name argument for WandB run name
                mode='online'
            )
        else:
            print("No internet, using wandb offline mode.")
            wandb.init(
                project='CCHRNET',
                config=opt,
                name=opt.name,  # Use the --name argument for WandB run name
                mode='offline'
            )
        wandb.watch(model, log='all')
    except:
        wandb.init(mode='offline')

    if opt.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        losses_cnt = AverageMeter()

        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.set_epoch(epoch)
        
        # Add progress bar for the inner loop
        with tqdm(total=len(dataset), desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}", unit="batch") as pbar:
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing

                # Use manual zero_grad, forward, compute_loss, etc.
                if opt.mixed_precision:
                    model.optimizer_G.zero_grad()
                    with torch.cuda.amp.autocast():
                        model.forward()
                        model.compute_loss()
                    scaler.scale(model.loss_all).backward()
                    scaler.step(model.optimizer_G)
                    scaler.update()
                else:
                    model.optimizer_G.zero_grad()
                    model.forward()
                    model.compute_loss()
                    model.loss_all.backward()
                    model.optimizer_G.step()

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)

                lss = 0.
                for k,v in model.get_current_losses().items():
                    lss += v
                losses_cnt.update(lss)
                
                # Log metrics to wandb
                if total_iters % opt.print_freq == 0:
                    current_lr = model.get_current_learning_rate()  # Assumes the model has this method
                    wandb.log({'loss': lss, 'learning_rate': current_lr, 'epoch': epoch, 'iter': total_iters})
                    # wandb.log({'loss': lss, 'epoch': epoch, 'iter': total_iters})
                # Update progress bar
                pbar.update(1)
        validate_model(opt, model, epoch, visualizer)
        if np.isnan(losses_cnt.avg):
            visualizer.print_msg("losses_cnt.avg is nan, jump out loop of epoch")
            break

        if losses_cnt.avg < best_loss:
            best_loss = losses_cnt.avg
            best_epoch = epoch
            model.save_networks("best")

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            visualizer.print_msg('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        losses = model.get_current_losses()
        losses['loss_avg'] = losses_cnt.avg
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, None, losses)
        visualizer.print_msg('End Epoch %d. Avg Loss: %f -------- ' % (epoch, losses_cnt.avg))
        visualizer.print_msg('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    wandb.finish()  # Optional: Finish the wandb run

    visualizer.print_msg('Finish Training. best_loss: %f, best_epoch: %d' % (best_loss, best_epoch))

    visualizer.print_msg('Train Done!')

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    print(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    train_model(opt, model)