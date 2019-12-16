from typing import List

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage


def show_debug_t(img_t: torch.Tensor, boxes_t: torch.Tensor):
    img: Image.Image = ToPILImage()(img_t)
    boxes = boxes_t.tolist()

    show_debug(img, boxes)


def show_debug(img: Image.Image, boxes: List[List[float]]):
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([int(x) for x in b], outline='red', width=2)
    img.show()
