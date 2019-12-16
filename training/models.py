import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes: int) -> FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model: FasterRCNN

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # 1 class for background

    return model
