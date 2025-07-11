import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image
import os.path
import pickle


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'dining table', 'dog', 'horse',
                     'motorbike', 'person', 'potted plant',
                     'sheep', 'sofa', 'train', 'tv']


def read_info_from_csv(path):
    infos = []
    df = pd.read_csv(path, dtype={"filename": np.str_, "labels": np.str_}, keep_default_na=False)
    df.apply(lambda x: infos.append(
        [x["filename"], list(filter(None, x["labels"].split("|")))]
    ), axis=1)
    return infos


class LvcitVoc(data.Dataset):
    def __init__(self, root, phase="predict", transform=None, target_transform=None, inp_name=None):
        self.root = root
        self.set = phase
        self.transform = transform
        self.target_transform = target_transform
        self.inp_name = inp_name
        self.classes = object_categories
        self.num_classes = len(object_categories)
        self.images = read_info_from_csv(os.path.join(root, "info.csv"))

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

        print('[dataset] VOC of LVCIT Set=%s number of classes=%d  number of images=%d' % (
            self.set, self.num_classes, len(self.images)
        ))

    def __getitem__(self, index):
        # e.g. filename: abc.jpg, labels: ["aeroplane", "bicycle"]
        filename, labels = self.images[index]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        target = np.zeros(self.num_classes, np.float32) - 1
        target[[object_categories.index(label) for label in labels]] = 1
        target = torch.from_numpy(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, filename, torch.tensor(self.inp)), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return self.num_classes

    def get_cat2id(self):
        return {label: i for i, label in enumerate(object_categories)}


class LvcitVoc2(LvcitVoc):
    def __init__(self, root, phase="predict", transform=None, target_transform=None, inp_name=None):
        super(LvcitVoc2, self).__init__(root, phase, transform, target_transform, inp_name)

    def __getitem__(self, index):
        # e.g. filename: abc.jpg, labels: ["aeroplane", "bicycle"]
        filename, labels = self.images[index]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        target = np.zeros(self.num_classes, np.float32)
        target[[object_categories.index(label) for label in labels]] = 1
        target = torch.from_numpy(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return filename, img, target
