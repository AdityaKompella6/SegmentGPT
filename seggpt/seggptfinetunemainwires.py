import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import sys


import random

import torchvision.transforms as T
import requests
import json
from io import BytesIO


def convert_segmentation_to_rgb(segmentation, category_colors):
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


def calculate_miou(mask_pred, mask_true):
    categories_pred = np.unique(mask_pred)
    categories_true = np.unique(mask_true)

    intersection = np.histogram2d(
        mask_pred.flatten(),
        mask_true.flatten(),
        bins=(categories_pred.shape[0], categories_true.shape[0]),
    )[0]

    area_pred = np.histogram(mask_pred, bins=categories_pred.shape[0])[0]
    area_true = np.histogram(mask_true, bins=categories_true.shape[0])[0]

    union = area_pred[:, None] + area_true - intersection

    iou = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union != 0
    )

    miou = np.mean(np.diag(iou))

    return miou




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


def create_contour(points, mask):
    # Create a blank image

    # Convert points to a numpy array
    points = np.array(points)

    # Reshape points to a 2D array
    points = points.reshape((-1, 1, 2))

    # Draw the contour
    cv2.drawContours(mask, [points], 0, 255, -1)


def construct_mask(img, annotations):
    temp_mask = np.zeros((img.size[1], img.size[0]))
    color = [255, 255, 255]
    for ann in annotations:
        if ann["label"] != "Wire":
            continue
        segs = ann["segmentation"]
        for seg in segs:
            points = seg["extPoints"]
            create_contour(points, temp_mask)
    mask = np.zeros((img.size[1], img.size[0], 3))
    mask[temp_mask == 255] = color
    mask = Image.fromarray(mask.astype(np.uint8)).convert("RGB")
    return mask

import numpy as np
from PIL import Image, ImageDraw

def generate_binary_mask(annotations, image_size):
    # Create a blank image
    image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        if ann["label"] != "Wire":
            continue
        segs = ann["segmentation"]
        for seg in segs:
            extPoints = seg["extPoints"]
            intPoints = seg["intPoints"]
            if len(extPoints) < 2 and len(intPoints) < 2:
                continue
            intPoints = [list(map(tuple,int)) for int in intPoints]
            extPoints = list(map(tuple,extPoints))
            # Draw the exterior contour
            draw.polygon(extPoints, outline=1, fill=1)

            # Draw the interior contours
            for interior_contour in intPoints:
                draw.polygon(interior_contour, outline=0, fill=0)

    # Convert the image to a binary mask (numpy array)
    binary_mask = np.array(image)
    mask = np.zeros((image_size[1], image_size[0], 3))
    color = [255, 255, 255]
    mask[binary_mask == 1] = color
    mask = Image.fromarray(mask.astype(np.uint8)).convert("RGB")

    return mask


with open("wires.json") as f:
    data = json.load(f)

url_set = set()
for image in data["images"]:
    url_set.add((image["url"], image["id"]))

image_set = set()
imageid_set = set()

for url, id in url_set:
    req = requests.get("http://" + url)
    img = Image.open(BytesIO(req.content)).convert("RGB").resize((448, 448))
    byte_img = img.tobytes()
    if byte_img not in image_set:
        image_set.add(byte_img)
        imageid_set.add(id)

##Some bad masks
# disclude_id_set = set(
#     [
#         "64696be14d61f800078e5be9",
#         "648b7c226f177e0007432879",
#         "646c4baeeee7ce000765e791",
#         "646c5784eee7ce00076678f9",
#         "646c6335eee7ce0007670362",
#         "646ce574eee7ce00076f6273",
#         "646ce65665bbde0007f09705",
#         "646ce5dce326cf0007b927f2",
#         "646d9cae09affa0007c9a1f0",
#         "646da0bf09affa0007c9cd35",
#         "646dbe8b65bbde0007fe0640",
#         "646df9dd09affa0007cea11e",
#         "646e15bfe326cf0007ca4d5f",
#         "646e185b4d61f80007ca4af1",
#         "646e185ae326cf0007ca7d3f",
#         "646e276109affa0007d1dab3",
#         "646e325809affa0007d2c5ec",
#         "646e361909affa0007d31a96",
#         "646e372b4d61f80007cc9555",
#         "646e37de09affa0007d341ca",
#         "646e37d609affa0007d340e0",
#         "646e3ea109affa0007d3cb65",
#         "646ebed7e326cf0007d4e660",
#         "646ed5ac4d61f80007d69cee",
#         "646f6a304d61f80007def5ca",
#         "647000e2cdccc800070edc63",
#         "647023199999350007945f9d",
#         "647058565ad8d50007eb67e9",
#         "64715fa42687e40007e9fa99",
#         "64716edb2687e40007eaa357",
#         "647179542687e40007eb1590",
#         "647185f2fa532900077861da",
#         "6471ae9a2687e40007eddabc",
#         "6471b3822687e40007ee10b3",
#         "6471c9892687e40007ef465c",
#         "6471e04ffa532900077c4ffc",
#         "6474cead50ebc70007df89d7",
#         "64755af42687e40007187283",
#     ]
# )

disclude_id_set = set(
    [
        "64696be14d61f800078e5be9",
        "648b7c226f177e0007432879",
        "646c4baeeee7ce000765e791",
        "646c5784eee7ce00076678f9",
        "646c6335eee7ce0007670362",
        "646ce574eee7ce00076f6273",
        "646ce65665bbde0007f09705",
        "646ce5dce326cf0007b927f2",
        "646d9cae09affa0007c9a1f0",
        "646da0bf09affa0007c9cd35",
        "646dbe8b65bbde0007fe0640",
        "646df9dd09affa0007cea11e",
        "646e15bfe326cf0007ca4d5f",
        "646e185b4d61f80007ca4af1",
        "646e185ae326cf0007ca7d3f",
        "646e276109affa0007d1dab3",
        "646e325809affa0007d2c5ec",
        "646e361909affa0007d31a96",
        "646e372b4d61f80007cc9555",
        "646e37de09affa0007d341ca",
        "646e37d609affa0007d340e0",
        "646e3ea109affa0007d3cb65",
        "646ebed7e326cf0007d4e660",
        "646ed5ac4d61f80007d69cee",
        "646f6a304d61f80007def5ca",
        "647000e2cdccc800070edc63",
        "647023199999350007945f9d",
        "647058565ad8d50007eb67e9",
        "64715fa42687e40007e9fa99",
        "64716edb2687e40007eaa357",
        "647179542687e40007eb1590",
        "647185f2fa532900077861da",
        "6471ae9a2687e40007eddabc",
        "6471b3822687e40007ee10b3",
        "6471c9892687e40007ef465c",
        "6471e04ffa532900077c4ffc",
        "6474cead50ebc70007df89d7",
        "64755af42687e40007187283",
        "64687349eee7ce00073655c8",
"646882634d61f8000783b0f1",
"646c3d0feee7ce00076544da",
"646c4c97eee7ce000765f48f",
"646c72a565bbde0007e89b6c",
"646c7346eee7ce000767cca7",
"646cc34deee7ce00076c53f3",
"646cc69ce326cf0007b6b0de",
"646cdb89eee7ce00076e65e5",
"646ce0e565bbde0007f02c55",
"646ce56eeee7ce00076f6223",
"646d991a65bbde0007fc2c0d",
"646da40465bbde0007fcaebc",
"646da6f509affa0007ca141a",
"646dc01509affa0007cb4e9c",
"6476d3f97d099e000720e9c9",
"646dcd5865bbde0007fee2d2",
"646de2af09affa0007cd4764",
"646de35165bbde0007002d58",
"646df3c365bbde0007012a84",
"646e058f09affa0007cf4fee",
"646e11db09affa0007d011a3",
"646e188e09affa0007d09cca",
"646ec2bb4d61f80007d5cefb",
"646ecb05e326cf0007d56e94",
"646ed5704d61f80007d69904",
"646f5cad65bbde00071876a7",
"646f713c65bbde00071a3273",
"647005228f02f700070dcc1c",
"64700117cdccc800070edfd4",
"647027d011dca20007a66a7a",
"64700133cdccc800070ee2cf",
"6470324c11dca20007a6e55a",
"6470592111dca20007a8a3e4",
"6470a16011dca20007ac705b",
"6470b42ed49b3a0007515686",
"6470da9cd49b3a00075373c7",
"64716676fa532900077701a1",
"6471744e2687e40007eae331",
"64756bb82687e400071917ef",
"646c64fceee7ce000767182a",
"646c654c4d61f80007b27c02",
"646de34d09affa0007cd5079",
"646defed4d61f80007c7e165",
"646e2e8f09affa0007d2760b",
"646eb85f65bbde00070ee93c",
"646ed60f65bbde000710303e",
"64704bbd9999350007962c15",
    ]
)

prompt_imgs = []
prompt_masks = []

for image in data["images"]:
    if image["id"] in imageid_set: #and image["id"] not in disclude_id_set:
        req = requests.get("http://" + image["url"])
        img = Image.open(BytesIO(req.content)).convert("RGB")
        prompt_imgs.append(img)
        # prompt_mask = construct_mask(img, image["tags"])
        prompt_mask = generate_binary_mask(image["tags"],img.size)
        prompt_masks.append(prompt_mask)


colors = [(0, 0, 0), (255, 255, 255)]
cat_to_color = {}
for i in range(2):
    cat_to_color[i] = colors[i]
color_to_cat = {v: k for (k, v) in cat_to_color.items()}


from seggptdataset import SegTensorDataset


ds = SegTensorDataset(prompt_imgs, prompt_masks, 0.2)


train_dl = DataLoader(ds, batch_size=2, pin_memory=True)

val_dl = DataLoader(ds, batch_size=1, pin_memory=True)


cat_to_class = {0: "Background", 1: "Wires"}

from seggptICLfinetune import fine_tune

from models_seggpt import LearnablePrompt

prompt = LearnablePrompt()

prompt.load_state_dict(torch.load("bestwires_learned_promptv3.pt"))

torch.cuda.empty_cache()

final_prompt, losses = fine_tune(
    train_dl,
    val_dl,
    color_to_cat,
    cat_to_class,
    prompt,
    "bestwires_learned_promptv4.pt",
    "seggpt-wires",
    1000,
    50,
    1e-4,
    1e-6,
    accum_iters=4,
)

torch.save(final_prompt.state_dict(), "wires_learned_promptv4.pt")
