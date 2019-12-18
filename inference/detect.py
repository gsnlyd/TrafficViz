from argparse import ArgumentParser
from argparse import ArgumentParser
from typing import NamedTuple, Tuple, List

import torch
import torchvision
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor

from training import models

COCO_LABELS_FILE_PATH = 'coco_labels.txt'
DEFAULT_DETECTION_THRESHOLD = 0.7

COCO_TRAFFIC_LABELS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    6: 'bus',
    8: 'truck'
}

LABEL_COLORS = {
    'person': 'pink',
    'bicycle': 'green',
    'car': 'lightblue',
    'motorcycle': 'orange',
    'bus': 'yellow',
    'truck': 'purple'
}
DEFAULT_COLOR = 'white'


class Detection(NamedTuple):
    box: Tuple[float, float, float, float]
    label: int
    label_str: str
    score: float


class Detector:
    def __init__(self, finetuned_path: str = None, threshold: float = DEFAULT_DETECTION_THRESHOLD,
                 filter_traffic_detections: bool = True):
        if finetuned_path is None:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = models.get_model(len(COCO_TRAFFIC_LABELS))
            self.model: FasterRCNN
            self.model.load_state_dict(torch.load(finetuned_path, map_location=torch.device('cpu')))

        self.model.eval()
        self.to_tensor = ToTensor()

        self.finetuned = finetuned_path is not None
        self.threshold = threshold
        self.filter_traffic_detections = not self.finetuned and filter_traffic_detections

        self.all_coco_labels = []
        if not self.filter_traffic_detections:
            with open(COCO_LABELS_FILE_PATH) as coco_file:
                for line in coco_file:
                    self.all_coco_labels.append(line.strip('\n'))

    def __label_properties(self, label: int) -> Tuple[int, str]:
        """Returns label and label str"""
        if not self.finetuned and not self.filter_traffic_detections:
            # Return properties for all COCO labels
            return label, self.all_coco_labels[label - 1]

        if not self.finetuned:
            # Re-map COCO labels to traffic labels
            label = list(COCO_TRAFFIC_LABELS.keys()).index(label) + 1

        return label, list(COCO_TRAFFIC_LABELS.values())[label - 1]

    def detect_objects(self, img: Image.Image) -> List[Detection]:
        img_tensor: Tensor = self.to_tensor(img).unsqueeze(0)  # Unsqueeze for a batch size of 1

        with torch.no_grad():
            result = self.model(img_tensor)[0]  # Result also has batch size of 1

        detections: List[Detection] = []
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            label = label.item()
            if score >= self.threshold and (not self.filter_traffic_detections or label in COCO_TRAFFIC_LABELS):
                box = tuple(box.tolist())
                score = score.item()

                label, label_str = self.__label_properties(label)
                detections.append(Detection(
                    box=box,
                    label=label,
                    label_str=label_str,
                    score=score
                ))

        return detections


def visualize(img: Image.Image, detections: List[Detection]):
    draw = ImageDraw.Draw(img)
    box_line_width = 3

    for d in detections:
        text = 'Label: {} ({}), Score: {:.2f}'.format(d.label_str, d.label, d.score)

        color = LABEL_COLORS[d.label_str] if d.label_str in LABEL_COLORS else DEFAULT_COLOR
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', '-i', type=str, required=True, help='Image on which to perform detection.')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_DETECTION_THRESHOLD,
                        help='Score threshold for detections.')
    parser.add_argument('--disable-traffic-filter', '-df', action='store_true',
                        help='Disable the filtering of unrelated categories.')
    parser.add_argument('--finetuned-path', '-f', type=str, default=None,
                        help='Path to load fine-tuned model weights.')

    args = parser.parse_args()
    print(args)

    detector = Detector(finetuned_path=args.finetuned_path,
                        threshold=args.threshold,
                        filter_traffic_detections=not args.disable_traffic_filter)

    im = Image.open(args.image_path)
    det = detector.detect_objects(im)
    for de in det:
        print(de)
    visualize(im, det)
    im.show()
