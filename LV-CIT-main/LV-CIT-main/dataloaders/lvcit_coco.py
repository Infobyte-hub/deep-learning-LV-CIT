import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image
import os.path
import pickle

cat2idx1 = {
    "airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball bat": 4,
    "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9,
    "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14,
    "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19,
    "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24,
    "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29,
    "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34,
    "hair dryer": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39,
    "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44,
    "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49,
    "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54,
    "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59,
    "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64,
    "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69,
    "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74,
    "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79
}
cat2idx2 = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
    'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
    'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
    'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
    'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40,
    'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
    'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56,
    'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
    'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70,
    'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77,
    'hair dryer': 78, 'toothbrush': 79
}
object_categories1 = list(cat2idx1.keys())
object_categories2 = list(cat2idx2.keys())
label_transform = {
    0: 4, 1: 47, 2: 24, 3: 46, 4: 34, 5: 35, 6: 21, 7: 59, 8: 13, 9: 1, 10: 14, 11: 8, 12: 73, 13: 39,
    14: 45, 15: 50, 16: 5, 17: 55, 18: 2, 19: 51, 20: 15, 21: 67, 22: 56, 23: 74, 24: 57, 25: 19, 26: 41,
    27: 60, 28: 16, 29: 54, 30: 20, 31: 10, 32: 42, 33: 29, 34: 23, 35: 78, 36: 26, 37: 17, 38: 52, 39: 66,
    40: 33, 41: 43, 42: 63, 43: 68, 44: 3, 45: 64, 46: 49, 47: 69, 48: 12, 49: 0, 50: 53, 51: 58, 52: 72,
    53: 65, 54: 48, 55: 76, 56: 18, 57: 71, 58: 36, 59: 30, 60: 31, 61: 44, 62: 32, 63: 11, 64: 28, 65: 37,
    66: 77, 67: 38, 68: 27, 69: 70, 70: 61, 71: 79, 72: 9, 73: 6, 74: 7, 75: 62, 76: 25, 77: 75, 78: 40, 79: 22
}


def read_info_from_csv(path):
    infos = []
    df = pd.read_csv(path, dtype={"filename": np.str_, "labels": np.str_}, keep_default_na=False)
    df.apply(lambda x: infos.append(
        [x["filename"], list(filter(None, x["labels"].split("|")))]
    ), axis=1)
    return infos


class LvcitCoco(data.Dataset):
    def __init__(self, root, phase="predict", transform=None, target_transform=None, inp_name=None):
        self.root = root
        self.set = phase
        self.transform = transform
        self.target_transform = target_transform
        self.inp_name = inp_name
        self.num_classes = len(object_categories1)
        self.images = read_info_from_csv(os.path.join(root, "info.csv"))

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] COCO of LVCIT Set=%s number of classes=%d  number of images=%d' % (
            self.set, self.num_classes, len(self.images)
        ))

    def __getitem__(self, index):
        # e.g. filename: abc.jpg, labels: ["aeroplane", "bicycle"]
        filename, labels = self.images[index]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        target = np.zeros(self.num_classes, np.float32) - 1
        target[[object_categories1.index(label) for label in labels]] = 1
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
        return cat2idx1


class LvcitCoco2(LvcitCoco):
    def __init__(self, root, phase="predict", transform=None, target_transform=None, inp_name=None):
        super(LvcitCoco2, self).__init__(root, phase, transform, target_transform, inp_name)

    def __getitem__(self, index):
        # e.g. filename: abc.jpg, labels: ["aeroplane", "bicycle"]
        filename, labels = self.images[index]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        target = np.zeros(self.num_classes, np.float32)
        target[[object_categories2.index(label) for label in labels]] = 1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return filename, img, target

    def get_cat2id(self):
        return cat2idx2
