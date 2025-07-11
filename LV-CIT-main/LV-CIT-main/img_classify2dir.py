import torch
from dataloaders.default_voc2 import Voc2007Classification2
from dataloaders.default_coco2 import COCO2014Classification2
import shutil
import os


root = os.path.join("data")
dst_dir = os.path.join("data", "combine", "0source_img")


if __name__ == "__main__":
    datasets = []
    voc_inp = os.path.join("data", "voc", "voc_glove_word2vec.pkl")
    val_dataset_voc = Voc2007Classification2(os.path.join(root, "voc"), phase="test", inp_name=voc_inp)
    datasets.append(("VOC", os.path.join(root, "voc", "VOCdevkit", "VOC2007", "JPEGImages"), val_dataset_voc))
    coco_inp = os.path.join("data", "coco", "coco_glove_word2vec.pkl")
    val_dataset_coco = COCO2014Classification2(os.path.join(root, "coco", "coco"), phase="val", inp_name=coco_inp)
    datasets.append(("COCO", os.path.join(root, "coco", "data", "val2014"), val_dataset_coco))
    for dataset, src_dir, dataloader in datasets:
        if not os.path.exists(os.path.join(dst_dir, dataset)):
            os.makedirs(os.path.join(dst_dir, dataset))
        cat2id = dataloader.get_cat2id()
        id2cat = list(cat2id.keys())
        for filename, _, target in dataloader:
            if not filename.endswith(".jpg"):
                filename = filename + ".jpg"
            print(filename)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            labels = [id2cat[i] for i in torch.nonzero(target).view(-1)]
            for label in labels:
                if not os.path.exists(os.path.join(dst_dir, dataset, label)):
                    os.makedirs(os.path.join(dst_dir, dataset, label))
                shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, dataset, label, filename))
