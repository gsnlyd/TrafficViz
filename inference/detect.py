from argparse import ArgumentParser
from typing import Optional, NamedTuple, Tuple, List

import torch
import torchvision
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import ToTensor

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

_model: Optional[Module] = None
_labels: Optional[List[str]] = None

img_transform = ToTensor()


def get_model() -> Module:
    global _model
    if _model is not None:
        return _model

    _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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


def visualize_detections(img: Image.Image, detections: List[Detection]):
    draw = ImageDraw.Draw(img)

    for d in detections:
        color = LABEL_COLORS[d.label] if d.label in LABEL_COLORS else 'white'

        draw.rectangle(d.box, outline=color, width=2)
        draw.text(d.box[0:2], text='Label: {} ({}), Score: {:.2f}'.format(d.label_str, d.label, d.score),
                  fill=color, stroke_width=1, stroke_fill='black')

    img.show()


def detect_objects(image_path: str, threshold: float = DEFAULT_DETECTION_THRESHOLD, disable_label_filter: bool = False,
                   print_detections: bool = False, visualize: bool = False) -> List[Detection]:
    img = Image.open(image_path)
    img_tensor: Tensor = img_transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Unsqueeze for a batch size of 1
    print(img_tensor.shape)

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

    if visualize:
        visualize_detections(img, detections)

    return detections


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', '-i', type=str, required=True, help='Image on which to perform detection.')
    parser.add_argument('--threshold', '-t', type=int, default=DEFAULT_DETECTION_THRESHOLD,
                        help='Score threshold for detections.')
    parser.add_argument('--disable-label-filter', '-df', action='store_true',
                        help='Disable the filtering of unrelated categories.')

    args = parser.parse_args()
    print(args)

    detect_objects(
        image_path=args.image_path,
        threshold=args.threshold,
        disable_label_filter=args.disable_label_filter,
        print_detections=True,
        visualize=True
    )
