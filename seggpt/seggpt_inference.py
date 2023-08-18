
import torch
import numpy as np
import sys
import time
from PIL import Image
import models_seggpt
import torch.nn.functional as F
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def generate_equally_spaced_colors(k):
    colors = []
    step = 360 / k  # Equally spaced hue step

    for i in range(k):
        hue = i * step  # Equally spaced hue values
        rgb = hsv_to_rgb(hue, 1, 1)  # Convert hue to RGB values
        scaled_rgb = tuple(
            int(val * 255) for val in rgb
        )  # Scale RGB values to 0-255 range
        colors.append(scaled_rgb)

    return colors


def hsv_to_rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        rgb = (c, x, 0)
    elif 60 <= h < 120:
        rgb = (x, c, 0)
    elif 120 <= h < 180:
        rgb = (0, c, x)
    elif 180 <= h < 240:
        rgb = (0, x, c)
    elif 240 <= h < 300:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)

    return tuple((val + m) for val in rgb)

def convert_segmentation_to_rgb(segmentation, category_colors):
    """Converts Mask where each entry is a unique integer corresponding 
    to class to the correct format"""
    height, width = segmentation.shape[:2]
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for category, color in category_colors.items():
        mask = segmentation == category
        rgb_mask[mask] = color

    return rgb_mask

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

ckpt_path = "./seggpt_vit_large.pth"
model = "seggpt_vit_large_patch16_input896x448"
device = "cuda"
seg_type = "instance"

device = "cuda"
device = torch.device(device)
model = prepare_model(ckpt_path, model, seg_type).to(device)
print("Model loaded.")



@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum("nhwc->nchw", x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum("nhwc->nchw", tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2 :] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)
    seg_type = torch.ones([valid.shape[0], 1])

    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(
        x.float().to(device),
        tgt.float().to(device),
        bool_masked_pos.to(device),
        valid.float().to(device),
        seg_type.to(device),
        feat_ensemble,
    )
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    output = y[0, y.shape[1] // 2 :, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def inference_image_custom(model, device, image, img2s, tgt2s):
    res, hres = 448, 448
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.0

    image_batch, target_batch = [], []
    for img2, tgt2 in zip(img2s, tgt2s):
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.0

        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.0

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)

        assert img.shape == (2 * res, res, 3), f"{img.shape}"
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2 * res, res, 3), f"{img.shape}"
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)

    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = (
        F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode="nearest",
        )
        .permute(0, 2, 3, 1)[0]
        .numpy()
    )
    output = output.astype(np.uint8)
    return output


device = "cuda"
ctx = torch.amp.autocast(device_type=device, dtype=torch.float16)
def inference_image_batch(model, device, prompt_img,prompt_mask,test_imgs):
    res, hres = 448, 448
    prompt_img = prompt_img.resize((res, hres))
    prompt_img = np.array(prompt_img)/ 255.0
    prompt_img = prompt_img - imagenet_mean
    prompt_img = torch.from_numpy(prompt_img / imagenet_std)
    prompt_mask = prompt_mask.resize((res, hres))
    prompt_mask = np.array(prompt_mask)/ 255.0
    prompt_mask = prompt_mask - imagenet_mean
    prompt_mask = torch.from_numpy(prompt_mask / imagenet_std)
    bs = len(test_imgs)
    image_batch = []
    for img in test_imgs:
        img = img.resize((res, hres))
        img = np.array(img) / 255.0
        img = img - imagenet_mean
        img = img / imagenet_std
        image_batch.append(torch.from_numpy(img))
    image_batch = torch.stack(image_batch, axis=0)
    prompt_img = prompt_img.repeat(bs, 1, 1, 1)
    prompt_mask = prompt_mask.repeat(bs, 1, 1, 1)
    input_img = torch.concatenate((prompt_img, image_batch), axis=1)
    target = torch.concatenate((prompt_mask, image_batch), axis=1)
    x = torch.einsum("nhwc->nchw", input_img)
    tgt = torch.einsum("nhwc->nchw", target)
    valid = torch.ones_like(tgt).float().to(device)
    seg_type = torch.zeros([valid.shape[0], 1]).to(device)
    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2 :] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).to(device)
    feat_ensemble = -1
    with torch.no_grad():
        with ctx:
                        loss, y, mask = model(
                            x.float().to(device),
                            tgt.float().to(device),
                            bool_masked_pos,
                            valid,
                            seg_type,
                            feat_ensemble,
                        )
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    output = y[:, y.shape[1] // 2 :, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    outputs = []
    for i in range(bs):
        size = test_imgs[i].size
        individual_output = (
        F.interpolate(
            output[i][None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode="nearest",
        )
        .permute(0, 2, 3, 1)[0]
        .numpy()
    )
        outputs.append(individual_output.astype(np.uint8))
    return outputs

def inference_image_batch_finetuned(model, device, prompt_img,prompt_mask,test_imgs):
    res, hres = 448, 448
    bs = len(test_imgs)
    image_batch = []
    for img in test_imgs:
        img = img.resize((res, hres))
        img = np.array(img) / 255.0
        img = img - imagenet_mean
        img = img / imagenet_std
        image_batch.append(torch.from_numpy(img))
    image_batch = torch.stack(image_batch, axis=0)
    prompt_img = prompt_img.repeat(bs, 1, 1, 1)
    prompt_mask = prompt_mask.repeat(bs, 1, 1, 1)
    input_img = torch.concatenate((prompt_img, image_batch), axis=1)
    target = torch.concatenate((prompt_mask, image_batch), axis=1)
    x = torch.einsum("nhwc->nchw", input_img)
    tgt = torch.einsum("nhwc->nchw", target)
    valid = torch.ones_like(tgt).float().to(device)
    seg_type = torch.zeros([valid.shape[0], 1]).to(device)
    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2 :] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).to(device)
    feat_ensemble = -1
    with torch.no_grad():
        with ctx:
                        loss, y, mask = model(
                            x.float().to(device),
                            tgt.float().to(device),
                            bool_masked_pos,
                            valid,
                            seg_type,
                            feat_ensemble,
                        )
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    output = y[:, y.shape[1] // 2 :, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    outputs = []
    for i in range(bs):
        size = test_imgs[i].size
        individual_output = (
        F.interpolate(
            output[i][None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode="nearest",
        )
        .permute(0, 2, 3, 1)[0]
        .numpy()
    )
        outputs.append(individual_output.astype(np.uint8))
    return outputs



def predict_batch(prompt_img,prompt_mask,test_imgs,threshold):
    """Runs inference for single test image, using SegGPT model

    Args:
        prompt_image (PIL Image): C x H x W tensor
        prompt_mask (np.array): H x W array where each element is an instance id
        test_imgs (list[PIL Image]): list[C x H x W tensor]
    """
    num_cats = len(np.unique(prompt_mask))
    colors = generate_equally_spaced_colors(num_cats)
    cat_to_color = {}
    for i in range(len(colors)):
        cat_to_color[i] = colors[i]
    color_to_cat = {v: k for (k, v) in cat_to_color.items()}
    prompt_mask = convert_segmentation_to_rgb(prompt_mask, cat_to_color)
    prompt_mask = Image.fromarray(prompt_mask).convert("RGB")
    outputs = inference_image_batch(
            model, device, prompt_img, prompt_mask, test_imgs
        )
    output_masks = []
    for output in outputs:
        output_mask = np.zeros((output.shape[0],output.shape[1]), dtype=np.uint8)
        for color, cat in color_to_cat.items():
            mask = get_color_mask(output, color, tolerance=threshold)
            
            output_mask[mask] = cat
        output_masks.append(output_mask)
        
    return output_masks,outputs

def predict_batch_finetuned(prompt_img,prompt_mask,test_imgs,threshold,num_cats):
    """Runs inference for single test image, using SegGPT model

    Args:
        prompt_image (PIL Image): C x H x W tensor
        prompt_mask (np.array): H x W array where each element is an instance id
        test_imgs (list[PIL Image]): list[C x H x W tensor]
        num_cats (int): Number of Semantic Categories
    """
    colors = generate_equally_spaced_colors(num_cats)
    cat_to_color = {}
    for i in range(len(colors)):
        cat_to_color[i] = colors[i]
    color_to_cat = {v: k for (k, v) in cat_to_color.items()}
    outputs = inference_image_batch_finetuned(
            model, device, prompt_img, prompt_mask, test_imgs
        )
    output_masks = []
    for output in outputs:
        output_mask = np.zeros((output.shape[0],output.shape[1]), dtype=np.uint8)
        for color, cat in color_to_cat.items():
            mask = get_color_mask(output, color, tolerance=threshold)
            
            output_mask[mask] = cat
        output_masks.append(output_mask)
        
    return output_masks,outputs



def predict(prompt_img,prompt_mask,test_img,threshold):
    """Runs inference for single test image, using SegGPT model

    Args:
        prompt_image (PIL Image): C x H x W tensor
        prompt_mask (np.array): H x W array where each element is an instance id
        test_img (PIL Image): C x H x W tensor
    """
    num_cats = len(np.unique(prompt_mask))
    colors = generate_equally_spaced_colors(num_cats)
    cat_to_color = {}
    for i in range(len(colors)):
        cat_to_color[i] = colors[i]
    color_to_cat = {v: k for (k, v) in cat_to_color.items()}
    prompt_mask = convert_segmentation_to_rgb(prompt_mask, cat_to_color)
    prompt_mask = Image.fromarray(prompt_mask).convert("RGB")
    output = inference_image_custom(
            model, device, test_img, [prompt_img], [prompt_mask]
        )
    output_mask = np.zeros((output.shape[0],output.shape[1]), dtype=np.uint8)
    for color, cat in color_to_cat.items():
        mask = get_color_mask(output, color, tolerance=threshold)
        
        output_mask[mask] = cat
        
    return output_mask,output

    
        
    
def split_image_into_grids(image, num_horizontal_cells, num_vertical_cells):
    width, height = image.size
    cell_width = width // num_horizontal_cells
    cell_height = height // num_vertical_cells
    grids = []
    grid_positions = []
    for y in range(0, num_vertical_cells):
        for x in range(0, num_horizontal_cells):
            grid_left = x * cell_width
            grid_right = (x + 1) * cell_width
            grid_top = y * cell_height
            grid_bottom = (y + 1) * cell_height
            
            if x == num_horizontal_cells - 1:
                grid_right = width
                
            if y == num_vertical_cells - 1:
                grid_bottom = height

            grid = image.crop((grid_left, grid_top, grid_right, grid_bottom))
            grids.append(grid)
            grid_positions.append((grid_left,grid_top))
    return grids,grid_positions


def stitch_masks(image_size, masks,grid_positions):
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for (x, y), m in zip(grid_positions,masks):
        mask[y:y + m.shape[0], x:x + m.shape[1]] = m

    return mask

import numpy as np
from skimage.measure import label, regionprops

def separate_masks(binary_mask, area_threshold):
    # Label connected components in the binary mask
    labeled_mask = label(binary_mask)
    
    # Get region properties of each connected component
    regions = regionprops(labeled_mask)
    
    # Initialize an empty list to store individual masks
    separate_masks = []
    
    # Iterate over each region and create a separate binary mask
    for region in regions:
        # Filter regions based on area threshold
        if region.area >= area_threshold:
            instance_mask = (labeled_mask == region.label).astype(np.uint8)
            separate_masks.append(instance_mask)
    
    return separate_masks


def combine_masks(masks):
    # Initialize an empty array to store the combined mask
    combined_mask = np.zeros_like(masks[0])

    # Assign unique instance IDs to each mask
    for i, mask in enumerate(masks, start=1):
        # Find the indices where the mask is True
        indices = np.where(mask == 1)

        # Assign the instance ID to those indices in the combined mask
        combined_mask[indices] = i

    return combined_mask


def total_segmented_area(mask):
    return np.sum(mask)

def predict_tiled(prompt_image,prompt_mask,test_image):
    image_size = test_image.size  
    ##Add Whole Image to Grids to be predicted on   
    test_grids = [test_image]
    
    num_horizontal_cells = 4
    num_vertical_cells = 4
    # Split the PIL image into grids
    grids4,grids_pos_4 = split_image_into_grids(test_image, num_horizontal_cells, num_vertical_cells)
    ##Add 4*4 Tiled Image to Grids to be predicted on   
    test_grids = test_grids + grids4
    
    num_horizontal_cells = 2
    num_vertical_cells = 2
    # Split the PIL image into grids
    grids2,grids_pos_2 = split_image_into_grids(test_image, num_horizontal_cells, num_vertical_cells)
    ##Add 2*2 Tiled Image to Grids to be predicted on 
    test_grids = test_grids + grids2
    # Predict instance segmentation masks for each grid
    masks,_ = predict_batch(prompt_image,prompt_mask,test_grids,400)
    # Stitch the masks together to create one segmentation mask
    result_mask4 = stitch_masks(image_size, masks[1:17], grids_pos_4)
    result_mask2 = stitch_masks(image_size, masks[17:], grids_pos_2)
    out_mask = masks[0]
    area_whole = total_segmented_area(out_mask)
    area_grid3 = total_segmented_area(result_mask2)
    area_grid4 = total_segmented_area(result_mask4)
    mask_levels = [out_mask,result_mask2,result_mask4]
    mask_areas = [area_whole,area_grid3,area_grid4]
    masks = separate_masks(mask_levels[np.argmax(mask_areas)],30)
    if len(masks) > 0:
        combined_mask = combine_masks(masks)
    else:
        combined_mask = None
        
    return combined_mask
    
    
def predict_tiled_finetuned(prompt_image,prompt_mask,test_image,seperate=False,num_cats=2):
    image_size = test_image.size  
    ##Add Whole Image to Grids to be predicted on   
    test_grids = [test_image]
    
    num_horizontal_cells = 4
    num_vertical_cells = 4
    # Split the PIL image into grids
    grids4,grids_pos_4 = split_image_into_grids(test_image, num_horizontal_cells, num_vertical_cells)
    ##Add 4*4 Tiled Image to Grids to be predicted on   
    test_grids = test_grids + grids4
    
    num_horizontal_cells = 2
    num_vertical_cells = 2
    # Split the PIL image into grids
    grids2,grids_pos_2 = split_image_into_grids(test_image, num_horizontal_cells, num_vertical_cells)
    ##Add 2*2 Tiled Image to Grids to be predicted on 
    test_grids = test_grids + grids2
    # Predict instance segmentation masks for each grid
    masks,_ = predict_batch_finetuned(prompt_image,prompt_mask,test_grids,400,num_cats)
    # Stitch the masks together to create one segmentation mask
    result_mask4 = stitch_masks(image_size, masks[1:17], grids_pos_4)
    result_mask2 = stitch_masks(image_size, masks[17:], grids_pos_2)
    out_mask = masks[0]
    area_whole = total_segmented_area(out_mask)
    area_grid3 = total_segmented_area(result_mask2)
    area_grid4 = total_segmented_area(result_mask4)
    mask_levels = [out_mask,result_mask2,result_mask4]
    mask_areas = [area_whole,area_grid3,area_grid4]
    if seperate:
        masks = separate_masks(mask_levels[np.argmax(mask_areas)],30)
        if len(masks) > 0:
            combined_mask = combine_masks(masks)
        else:
            combined_mask = None       
        return combined_mask
    else:
        # return mask_levels,np.argmax(mask_areas)
        return mask_levels[np.argmax(mask_areas)]
