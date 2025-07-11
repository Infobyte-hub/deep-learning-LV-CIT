import argparse
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import shutil
import glob
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tqdm

from lvcit_main import model_info
from lvcit_runner import get_model, create_engine
from models import asl_validate_multi


object_categories = {
    "voc": [[
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'dining table', 'dog', 'horse',
        'motorbike', 'person', 'potted plant',
        'sheep', 'sofa', 'train', 'tv'
    ]],
    "coco": [
        [
            "airplane", "apple", "backpack", "banana", "baseball bat",
            "baseball glove", "bear", "bed", "bench", "bicycle",
            "bird", "boat", "book", "bottle", "bowl",
            "broccoli", "bus", "cake", "car", "carrot",
            "cat", "cell phone", "chair", "clock", "couch",
            "cow", "cup", "dining table", "dog", "donut",
            "elephant", "fire hydrant", "fork", "frisbee", "giraffe",
            "hair dryer", "handbag", "horse", "hot dog", "keyboard",
            "kite", "knife", "laptop", "microwave", "motorcycle",
            "mouse", "orange", "oven", "parking meter", "person",
            "pizza", "potted plant", "refrigerator", "remote", "sandwich",
            "scissors", "sheep", "sink", "skateboard", "skis",
            "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
            "surfboard", "teddy bear", "tennis racket", "tie", "toaster",
            "toilet", "toothbrush", "traffic light", "train", "truck",
            "tv", "umbrella", "vase", "wine glass", "zebra",
        ],
        [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"
        ]
    ]
}
matting_img_root = os.path.join("data", "lvcit", "2matting_img")


class ObjectLibrary(data.Dataset):
    def __init__(self, root, dataname, transform=None, target_transform=None, inp_name=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.inp_name = inp_name
        self.dataname = dataname
        self.classes = object_categories[dataname][0]
        self.num_classes = len(self.classes)
        self.images = self.read_images()

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

        print('[dataset] ObjectLibrary of %s, number of classes=%d,  number of images=%d' % (
            self.dataname, self.num_classes, len(self.images)
        ))

    def read_images(self):
        images = []
        for label in self.classes:
            for img in os.listdir(os.path.join(self.root, label)):
                images.append((img, label))
        return images

    def __getitem__(self, index):
        img_name, label = self.images[index]
        try:
            img = Image.open(os.path.join(self.root, label, img_name)).convert('RGB')
        except:
            raise Exception(os.path.join(self.root, label, img_name))
        target = np.zeros(self.num_classes, np.float32) - 1
        target[self.classes.index(label)] = 1
        target = torch.from_numpy(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, img_name, torch.tensor(self.inp)), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return self.num_classes

    def get_cat2id(self):
        return {label: i for i, label in enumerate(self.classes)}


class ObjectLibrary2(ObjectLibrary):
    def __init__(self, root, dataname, transform=None, target_transform=None, inp_name=None):
        super(ObjectLibrary2, self).__init__(root, dataname, transform, target_transform, inp_name)
        self.classes = object_categories[dataname][1 % len(object_categories[dataname])]

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = Image.open(os.path.join(self.root, label, img_name)).convert('RGB')
        target = np.zeros(self.num_classes, np.float32)
        target[self.classes.index(label)] = 1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_name, img, target


def check(datasets=None):
    if datasets is None:
        datasets = ["voc", "coco"]
    model_info_index = {
        "voc": [0, 1, 2],
        "coco": [0, 1, 2],
    }
    num_classes = {
        "voc": 20,
        "coco": 80
    }
    special_info = {
        "coco_msrn": {
            "pool_ratio": 0.05,
            "batch_size": 16,
        },
        "coco_mlgcn": {
            "batch_size": 16,
        },
        "coco_asl": {
            "workers": 1,
            "model_type": "tresnet_l",
            "print_freq": 64,
            "model_path": os.path.join("checkpoints", "asl", "coco_checkpoint.pth"),
            "batch_size": 16,
        },
    }
    for dataset in datasets:
        res_df = pd.DataFrame()
        for idx in model_info_index[dataset]:
            print(model_info[idx]["model_name"])
            info = {
                **model_info[idx],
                "resume": glob.glob(os.path.join("checkpoints", model_info[idx]["model_name"], f"{dataset}_checkpoint.*"))[0],
                "data_name": dataset,
                "num_classes": num_classes[dataset],
                "graph_file": os.path.join("data", dataset, f"{dataset}_adj.pkl"),
                "use_gpu": torch.cuda.is_available(),
                **(special_info[f"{dataset}_{model_info[idx]['model_name']}"] if f"{dataset}_{model_info[idx]['model_name']}" in special_info else {}),
            }
            args = argparse.Namespace(**info)
            model_class = get_model(args.model_name)
            if args.model_name in ['msrn', 'mlgcn']:
                lib_dataset = ObjectLibrary(
                    os.path.join(matting_img_root, f"{dataset.upper()}_output"),
                    dataset,
                    inp_name=os.path.join("data", dataset, f"{dataset}_glove_word2vec.pkl")
                )
                model, criterion, optimizer, engine = create_engine(model_class, args)
                engine.predict(model, criterion, lib_dataset, optimizer)

                pb = torch.sigmoid(engine.state['ap_meter'].scores)
                result = []
                for i in range(len(pb)):
                    temp = {"filename": engine.state['names'][i]}
                    pred_labels = np.where(pb.numpy()[i] > args.threshold, 1, 0)
                    gt_label_idx = np.where(engine.state['ap_meter'].targets[i] == 1)[0][0]
                    success = sum(pred_labels) == 1 & pred_labels[gt_label_idx]
                    temp["target"] = lib_dataset.classes[gt_label_idx]
                    temp[args.model_name] = int(success)
                    result.append(temp)
            elif args.model_name in ['asl']:
                transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ])
                validate = asl_validate_multi
                model = model_class(args)
                lib_dataset = ObjectLibrary2(
                    os.path.join(matting_img_root, f"{dataset.upper()}_output"),
                    dataset,
                    inp_name=os.path.join("data", dataset, f"{dataset}_glove_word2vec.pkl"),
                    transform=transform,
                )
                lib_loader = torch.utils.data.DataLoader(
                    lib_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True
                )
                results, _, targets = validate(lib_loader, model, args)
                results["target"] = pd.DataFrame(targets).apply(
                    lambda x: "|".join([lib_dataset.classes[i] for i, v in enumerate(x) if v == 1]),
                    axis=1
                )
                results[args.model_name] = results[results.columns.difference(["img"])].apply(
                    lambda x: int((len(list(filter(None, x.tolist()))) == 2) & (x[1] == x["target"])),
                    axis=1
                )
                result = results[["img", "target", args.model_name]].rename(columns={"img": "filename"})
            result = pd.DataFrame(result)
            result["tmp_key"] = result[["filename", "target"]].apply(
                lambda x: x["filename"] + "-" + x["target"],
                axis=1
            )
            if res_df.empty:
                res_df = result
            else:
                res_df = pd.merge(res_df, result[[args.model_name, "tmp_key"]], on="tmp_key")
        res_df.drop(columns=["tmp_key"], inplace=True)
        print(res_df)
        res_df.to_csv(os.path.join(
            matting_img_root, f"{dataset.upper()}_output", "object_detect.csv"
        ), index=False)


def model_select(datasets=None):
    if datasets is None:
        datasets = ["voc", "coco"]
    for dataset in datasets:
        src_dir = os.path.join(matting_img_root, f"{dataset.upper()}_output")
        dst_dir = os.path.join(matting_img_root, f"{dataset.upper()}_output_model_pass")
        res_file = os.path.join(src_dir, "object_detect.csv")
        res_df = pd.read_csv(res_file)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for target in res_df["target"].unique():
            if not os.path.exists(os.path.join(dst_dir, target)):
                os.mkdir(os.path.join(dst_dir, target))
        # select images that at least one model pass
        res_df["success"] = res_df.apply(lambda x: any(x[2:]), axis=1)
        res_df = res_df[res_df["success"]]
        tqdm.tqdm.pandas(desc=f"Copying files of {dataset}...")
        res_df.progress_apply(
            lambda x: shutil.copy(
                os.path.join(src_dir, x["target"], x["filename"]),
                os.path.join(dst_dir, x["target"], x["filename"])
            ),
            axis=1
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check object library by DNN")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="select dataset, default both voc and coco"
    )
    args = parser.parse_args()

    if args.dataset is None:
        check()
        model_select()
    else:
        check([args.dataset])
        model_select([args.dataset])
