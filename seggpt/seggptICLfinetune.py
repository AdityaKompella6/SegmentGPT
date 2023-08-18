##Using mixed precision float 16
##Batch Size of 2 is max so using grad accumulation so that all the few shot samples are in one batch
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import models_seggpt
from torch.utils.data import DataLoader, Dataset
import wandb
import random
import bitsandbytes as bnb


def prepare_model(
    chkpt_dir, arch="seggpt_vit_large_patch16_input896x448", seg_type="instance"
):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


res, hres = 448, 448


import math


def adjust_learning_rate(optimizer, epoch, lr, warmup_epochs, min_lr, total_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_color_mask(image, target_color, tolerance=50):
    # Convert target color to numpy array
    target_color = np.array(target_color)

    # Convert image to numpy array
    image_array = np.array(image)

    # Calculate the color distance between each pixel and the target color
    color_distance = np.abs(image_array - target_color)

    # Sum the color distance along the RGB channels
    color_distance_sum = np.sum(color_distance, axis=2)

    # Create a binary mask where pixels with small color distances are True
    mask = color_distance_sum <= tolerance

    return mask


def calculate_miou(pred, gt):

  classes = np.unique(gt)
  classes = classes[classes != 0]
  if len(classes) == 0:
    return 0 # or nan/other default value
  iou_list = []

  for c in classes:
    pred_c = (pred == c)   
    gt_c = (gt == c)

    intersection = np.logical_and(pred_c, gt_c).sum()
    union = np.logical_or(pred_c, gt_c).sum()
    eps = 1e-6
    iou = intersection / (union + eps)
    iou_list.append(iou)

  miou = np.mean(iou_list)
  return miou 

def fine_tune(
    train_dataloader,
    val_dataloader,
    color_to_cat,
    cat_to_class_name,
    prompt,
    chkpt_name,
    project_name,
    epochs=200,
    warmup_epochs=50,
    max_lr=3e-3,
    min_lr=3e-5,
    accum_iters=3,
):
    """Fine tunes model on few_shot datasetusing In-Context Fine-Tuning

    Args:
        train_dataloader (dataloader): N x H x W x 3 (image, gt_mask) dataloader
        batch_size (int): batch_size for training
        epochs (int): How many epochs of training
        max_lr (float): The maximum lr during training
        min_lr (float): The minimum lr during training
    """
    ckpt_path = "./seggpt_vit_large.pth"
    model = "seggpt_vit_large_patch16_input896x448"
    device = "cuda"
    seg_type = "semantic"
    device = "cuda"
    ctx = torch.amp.autocast(device_type=device, dtype=torch.float16)
    ctx_val = torch.amp.autocast(device_type=device, dtype=torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = prepare_model(ckpt_path, model, seg_type).to(device)
    print("Model loaded.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            count += parameter.numel()

    print(f"Trainable Params: {count}")
    best_avg_loss = float("inf")
    optimizer = bnb.optim.AdamW8bit(prompt.parameters(), lr=3e-4, weight_decay=0.075)
    epoch_losses = []
    feat_ensemble = -1
    prompt = prompt.to(device)
    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2 :] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    bool_masked_pos = bool_masked_pos.to(device)

    wandb.init(project=project_name)
    for epoch in range(epochs):
        losses = []
        batch_loss = 0
        for batch_idx, (train_img, train_mask) in enumerate(train_dataloader):
            bs = train_img.shape[0]
            train_img, train_mask = train_img.cuda(non_blocking=True), train_mask.cuda(
                non_blocking=True
            )
            img_prompt, mask_prompt = prompt()
            img_prompt = img_prompt.repeat(bs, 1, 1, 1)
            mask_prompt = mask_prompt.repeat(bs, 1, 1, 1)
            input_img = torch.concatenate((img_prompt, train_img), axis=1)
            target = torch.concatenate((mask_prompt, train_mask), axis=1)
            x = torch.einsum("nhwc->nchw", input_img)
            tgt = torch.einsum("nhwc->nchw", target)
            valid = torch.ones_like(tgt).float().to(device)
            seg_type = torch.zeros([valid.shape[0], 1]).to(device)
            with ctx:
                loss, y, mask = model(
                    x,
                    tgt,
                    bool_masked_pos,
                    valid,
                    seg_type,
                    feat_ensemble,
                )
            loss = loss / accum_iters
            scaler.scale(loss).backward()
            batch_loss += loss.item()
            if ((batch_idx + 1) % accum_iters == 0) or (
                batch_idx + 1 == len(train_dataloader)
            ):
                if (batch_idx + 1) % accum_iters == 0:
                    losses.append(batch_loss)
                    wandb.log({"Loss": batch_loss})
                batch_loss = 0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if epoch % 2 == 0:
            avg_loss = np.mean(losses)
            if avg_loss < best_avg_loss:
                torch.save(prompt.state_dict(), chkpt_name)
                best_avg_loss = avg_loss
            print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
            wandb.log({"Epoch Loss": np.mean(losses)})
            epoch_losses.append(np.mean(losses))

        if epoch % 5 == 0:
            mIoUs = []
            wb_masks = []
            val_losses = []
            for val_img, val_mask in val_dataloader:
                bs = val_img.shape[0]
                val_img, val_mask = val_img.cuda(non_blocking=True), val_mask.cuda(
                    non_blocking=True
                )
                img_prompt, mask_prompt = prompt()
                img_prompt = img_prompt.repeat(bs, 1, 1, 1)
                mask_prompt = mask_prompt.repeat(bs, 1, 1, 1)
                input_img = torch.concatenate((img_prompt, val_img), axis=1)
                target = torch.concatenate((mask_prompt, val_mask), axis=1)
                x = torch.einsum("nhwc->nchw", input_img)
                tgt = torch.einsum("nhwc->nchw", target)
                valid = torch.ones_like(tgt).float().to(device)
                seg_type = torch.zeros([valid.shape[0], 1]).to(device)
                with ctx_val:
                    with torch.no_grad():
                        loss, y, mask = model(
                            x,
                            tgt,
                            bool_masked_pos,
                            valid,
                            seg_type,
                            feat_ensemble,
                        )
                val_losses.append(loss.item())
                y = model.unpatchify(y)
                y = torch.einsum("nchw->nhwc", y).detach().cpu()
                output = y[0, y.shape[1] // 2 :, :, :]
                output = torch.clip(
                    (output * imagenet_std + imagenet_mean) * 255, 0, 255
                )
                gt_mask = torch.clip(
                    (val_mask.cpu() * imagenet_std + imagenet_mean) * 255, 0, 255
                ).reshape(448, 448, 3)
                processed_val_image = np.array(
                    (
                        torch.clip(
                            (val_img.cpu() * imagenet_std + imagenet_mean) * 255, 0, 255
                        ).reshape(448, 448, 3)
                    )
                ).astype(np.uint8)
                output_mask = np.zeros((448, 448), dtype=np.uint8)
                ouput_gt_mask = np.zeros((448, 448), dtype=np.uint8)
                for color, cat in color_to_cat.items():
                    mask = get_color_mask(output, color, tolerance=150)
                    mask_gt = get_color_mask(gt_mask, color, tolerance=10)
                    output_mask[mask] = cat
                    ouput_gt_mask[mask_gt] = cat
                mIoUs.append(calculate_miou(output_mask, ouput_gt_mask))
                if random.random() < 0.02:
                    wb_mask = wandb.Image(
                        processed_val_image,
                        masks={
                            "prediction": {
                                "mask_data": output_mask,
                                "class_labels": cat_to_class_name,
                            },
                            "ground truth": {
                                "mask_data": ouput_gt_mask,
                                "class_labels": cat_to_class_name,
                            },
                        },
                    )
                    wb_masks.append(wb_mask)
            wandb.log({"Val Mean mIoU": np.mean(mIoUs)})
            wandb.log({"Val Loss": np.mean(val_losses)})
            if len(wb_masks) > 0:
                wandb.log({"predictions": wb_masks})
        lr = adjust_learning_rate(
            optimizer, epoch, max_lr, warmup_epochs, min_lr, epochs
        )
        wandb.log({"lr": lr})
    return prompt, epoch_losses
