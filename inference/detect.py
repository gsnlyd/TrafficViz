import time
from argparse import ArgumentParser
from typing import Optional, NamedTuple, Tuple, List, Union

import torch
import torchvision
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import Module
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor

from training import models

DEFAULT_DETECTION_THRESHOLD = 0.7
ALLOWED_LABELS = [
    1,  # Person
    2,  # Bicycle
    3,  # Car
    4,  # Motorcycle
    6,  # Bus
    8,  # Truck
]

LABEL_COLORS = {
    1: 'pink',
    2: 'green',
    3: 'lightblue',
    4: 'orange',
    6: 'yellow',
    8: 'purple'
}

_model: Optional[FasterRCNN] = None
_labels: Optional[List[str]] = None

img_transform = ToTensor()


def get_model(load_finetuned: str = None) -> Module:
    global _model
    if _model is not None:
        return _model

    if load_finetuned is None:
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        _model = models.get_model(len(ALLOWED_LABELS))
        _model.load_state_dict(torch.load(load_finetuned, map_location=torch.device('cpu')))

    _model.eval()

    return _model


def get_label_str(label: int) -> str:
    global _labels
    if _labels is None:
        _labels = []
        with open('coco_labels.txt') as labels_file:
            for l in labels_file:
                label_str = l.strip('\n')
                _labels.append(label_str)

    return _labels[label - 1]


class Detection(NamedTuple):
    box: Tuple[int, int, int, int]
    label: int
    label_str: str
    score: float


def visualize_detections(img: Image.Image, detections: List[Detection], show: bool = False) -> Image.Image:
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img)

    box_line_width = 3

    for d in detections:
        color = LABEL_COLORS[d.label] if d.label in LABEL_COLORS else 'white'
        text = 'Label: {} ({}), Score: {:.2f}'.format(d.label_str, d.label, d.score)

        draw.rectangle(d.box, outline=color, width=box_line_width)

        text_width, text_height = draw.textsize(text)

        text_coords = (d.box[0] + box_line_width + 1,
                       d.box[1] + box_line_width + 1)
        text_background_box = (
            text_coords[0] - 1,
            text_coords[1] - 1,
            text_coords[0] + text_width,
            text_coords[1] + text_height
        )

        draw.rectangle(text_background_box, fill='black')
        draw.text(text_coords, text=text, fill=color)

    if show:
        viz_img.show()
    return viz_img


def detect_objects(image_path: str, threshold: float = DEFAULT_DETECTION_THRESHOLD, disable_label_filter: bool = False,
                   print_detections: bool = False, visualize: bool = False, show_visualization: bool = False) -> \
        Tuple[List[Detection], Optional[Image.Image]]:
    start_time = time.time()

    img = Image.open(image_path)
    img_tensor: Tensor = img_transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Unsqueeze for a batch size of 1

    model = get_model()
    with torch.no_grad():
        result = model(img_tensor)[0]  # Result also has batch size of 1

    detections: List[Detection] = []

    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
        if score < threshold or (not disable_label_filter and label not in ALLOWED_LABELS):
            continue

        box = (
            box[0].item(),
            box[1].item(),
            box[2].item(),
            box[3].item()
        )

        label = label.item()
        score = score.item()

        detections.append(Detection(
            box=box,
            label=label,
            label_str=get_label_str(label),
            score=score
        ))

    if print_detections:
        for d in detections:
            print(d)
        print('Detection took {:.2f} seconds'.format(time.time() - start_time))

    if visualize:
        viz_img = visualize_detections(img, detections, show=show_visualization)
        return detections, viz_img

    return detections, None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', '-i', type=str, required=True, help='Image on which to perform detection.')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_DETECTION_THRESHOLD,
                        help='Score threshold for detections.')
    parser.add_argument('--disable-label-filter', '-df', action='store_true',
                        help='Disable the filtering of unrelated categories.')
    parser.add_argument('--finetuned-path', '-f', type=str, default=None,
                        help='Path to load fine-tuned model weights.')

    args = parser.parse_args()
    print(args)

    if args.finetuned_path is not None:
        get_model(args.finetuned_path)

    detect_objects(
        image_path=args.image_path,
        threshold=args.threshold,
        disable_label_filter=args.disable_label_filter,
        print_detections=True,
        visualize=True,
        show_visualization=True
    )
