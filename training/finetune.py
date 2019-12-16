import os
from typing import List, Dict

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor

from training import models

MODELS_DIR = 'model_parameters'

REQUIRE_CATEGORIES = [3, 8]

USE_LABELS = [
    1,  # Person
    2,  # Bicycle
    3,  # Car
    4,  # Motorcycle
    6,  # Bus
    8,  # Truck
]


class CocoSubset(VisionDataset):
    def __init__(self, images_dir: str, annotations_path: str,
                 img_categories: List[int], ann_categories: List[int],
                 transform=None):
        super(CocoSubset, self).__init__(images_dir, None, transform, None)
        self.coco = COCO(annotations_path)
        self.img_ids = self.coco.getImgIds(catIds=img_categories)

        self.ann_categories = ann_categories

    def __len__(self):
        return len(self.img_ids)

    def __map_category(self, category: int) -> int:
        return self.ann_categories.index(category)

    def __getitem__(self, item):
        img_id = self.img_ids[item]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.ann_categories, iscrowd=False)

        img_name = self.coco.loadImgs(ids=img_id)[0]['file_name']
        raw_img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        raw_target = self.coco.loadAnns(ann_ids)
        if self.transform is not None:
            img = self.transform(raw_img)
        else:
            img = raw_img

        target = {
            'boxes': [],
            'labels': []
        }

        for d in raw_target:
            b = d['bbox']
            target['boxes'].append((
                b[0],
                b[1],
                b[0] + b[2],
                b[1] + b[3]  # Convert from (x y w h) to (x1 y1 x2 y2)
            ))
            cat = self.ann_categories.index(d['category_id'])  # Re-compute category id from subset categories
            target['labels'].append(cat)

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        return img, target


def batch_collate(x):
    x = tuple(zip(*x))
    x = list(x[0]), list(x[1])
    return x


class Metrics:
    def __init__(self):
        self.m_dict = {}

    def log(self, d: Dict[str, torch.Tensor]):
        for k, v in d.items():
            if k not in self.m_dict:
                self.m_dict[k] = []
            self.m_dict[k].append(v.item())

    def avg_dict(self, last: int = None) -> Dict[str, float]:
        a_dict = {}
        if last is None:
            it = list(self.m_dict.items())
        else:
            last = min(len(self.m_dict), last)
            it = list(self.m_dict.items())[-last:]
        for k, v in it:
            a_dict[k] = sum(v) / len(v)
        return a_dict


def save_parameters(save_dir: str, model: FasterRCNN, epoch: int):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), save_path)
    print('Saved model at path {}'.format(save_path))


def train(epochs: int = 100, batch_size: int = 1):
    model: FasterRCNN = models.get_model(len(USE_LABELS))
    train_dataset = CocoSubset(images_dir='datasets/train2017',
                               annotations_path='datasets/annotations/instances_train2017.json',
                               img_categories=[3, 8], ann_categories=USE_LABELS,
                               transform=ToTensor())
    val_dataset = CocoSubset(images_dir='datasets/val2017',
                             annotations_path='datasets/annotations/instances_val2017.json',
                             img_categories=[3, 8], ann_categories=USE_LABELS,
                             transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=batch_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=batch_collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    for epoch_i in range(epochs):
        for mode, loader in loaders.items():
            print()

            m = Metrics()
            with torch.set_grad_enabled(mode == 'train'):
                for batch_i, (img, target) in enumerate(loader):
                    img = [i.to(device) for i in img]
                    target = [{k: t.to(device) for k, t in d.items()} for d in target]

                    losses_dict: Dict[str, torch.Tensor] = model(img, target)
                    total_loss = sum(list(losses_dict.values()))

                    m.log(losses_dict)
                    m.log({'total_loss': total_loss})

                    if mode == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    if batch_i % (len(loader) / 10) == 0:
                        print('epoch={} mode={} batch={}/{} --- '.format(
                            epoch_i + 1,
                            mode,
                            batch_i + 1,
                            len(loader)
                        ) + ', '.join(['{}={}'.format(k, v) for k, v in m.avg_dict().items()]))

        scheduler.step(epoch_i)
        save_parameters(MODELS_DIR, model, epoch_i + 1)
        print('\n\n')


if __name__ == '__main__':
    train()
