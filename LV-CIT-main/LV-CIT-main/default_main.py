import traceback
import warnings
import time
import argparse
import os
import joblib
import glob
import pandas as pd

from util import cal_score
from default_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')


# data info
root = "./data"
data_info = [
    {
        "data_name": "voc",
        "data": os.path.join(root, "voc"),
        "phase": "test",
        "num_classes": 20,
        "res_path": os.path.join(root, "voc", "results"),
        "inp_name": "data/voc/voc_glove_word2vec.pkl",
        "graph_file": "data/voc/voc_adj.pkl",
    },
    {
        "data_name": "coco",
        "data": os.path.join(root, "coco", "coco"),
        "phase": "val",
        "num_classes": 80,
        "res_path": os.path.join(root, "coco", "results"),
        "inp_name": "data/coco/coco_glove_word2vec.pkl",
        "graph_file": "data/coco/coco_adj.pkl",
    },
]

# model info
model_info = [
    # 0 msrn
    {
        "model_name": "msrn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/msrn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "pretrained": 1,
        "pretrain_model": "checkpoints/msrn/resnet101_for_msrn.pth.tar",
        "pool_ratio": 0.2,
        "backbone": "resnet101",
        "save_model_path": "checkpoints/save/msrn",
    },
    # 1 ml gcn
    {
        "model_name": "mlgcn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/ml_gcn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "save_model_path": "checkpoints/save/mlgcn",
    },
    # 3 asl
    {
        "model_name": "asl",
        "model_type": "tresnet_xl",
        "model_path": "checkpoints/asl/voc_checkpoint.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 8,
        "print_freq": 64,
    },
]

TASKS = [
    # voc msrn
    {
        "task_name": "voc_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader": Voc2007Classification,
    },
    # voc mlgcn
    {
        "task_name": "voc_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader": Voc2007Classification,
    },
    # voc asl
    {
        "task_name": "voc_asl",
        "args": {**data_info[0], **model_info[2]},
        "dataloader": Voc2007Classification2,
    },

    # coco msrn
    {
        "task_name": "coco_msrn",
        "args": {
            **data_info[1], **model_info[0],
            "pool_ratio": 0.05,
            "resume": "checkpoints/msrn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco mlgcn
    {
        "task_name": "coco_mlgcn",
        "args": {
            **data_info[1], **model_info[1],
            "resume": "checkpoints/ml_gcn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco asl
    {
        "task_name": "coco1_asl",
        "args": {
            **data_info[1], **model_info[2],
            "batch_size": 1,
            "model_type": "tresnet_l",
            "model_path": "checkpoints/asl/coco_checkpoint.pth",
            "part": 10,
            "print_freq": 640,
        },
        "dataloader": COCO2014Classification2,
    },
]


def randomly_select(select_num, repeat):
    VERSION = "_v6_random_255_255_255_255_s1a0"
    dst_root = os.path.join(root, "lvcit", "4results")
    ca_root = os.path.join(root, "lvcit", "1covering_array")
    dst_dir_name = {
        "voc": "VOC_20",
        "coco": "COCO_80"
    }
    ca_types = [f"_{k}_{tau}" for k in [4] for tau in [2]]
    num_classes = {
        "voc": 20,
        "coco": 80
    }
    models = [
        ("msrn", "voc"),
        ("mlgcn", "voc"),
        ("asl", "voc"),
        ("msrn", "coco"),
        ("mlgcn", "coco"),
        ("asl", "coco"),
    ]
    cat2idxes = {
        "voc": {
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
            'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
            'cow': 9, 'dining table': 10, 'dog': 11, 'horse': 12,
            'motorbike': 13, 'person': 14, 'potted plant': 15,
            'sheep': 16, 'sofa': 17, 'train': 18, 'tv': 19
        },
        "coco": {
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
    }
    for i in range(repeat):
        selected = {}
        for model, data in models:
            res_df = pd.read_excel(os.path.join(root, data, "results", f"result_{data}_{model}.xlsx")).fillna("")
            res_df["pass"] = res_df.apply(
                lambda x: 1 if x["labels_gt"] == x["labels"] else 0, axis=1
            )
            res_df = res_df[['filename', 'labels_gt', 'labels', 'pass']]
            for ca_type in ca_types:
                ca_type = f"adaptive random_{num_classes[data]}" + ca_type
                dst_dir = os.path.join(dst_root, dst_dir_name[data] + VERSION, "random", f"{ca_type}_No{i+1}")
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                print()
                ca_path = glob.glob(os.path.join(ca_root, ca_type.split("_")[0], f"ca_{ca_type}*.csv"))[i]
                ca_df = pd.read_csv(ca_path)
                img_num = len(ca_df) * select_num
                if f"{data}_{ca_type}" not in selected:
                    res = res_df.sample(img_num).reset_index(drop=True)
                    selected[f"{data}_{ca_type}"] = res[["filename"]]
                else:
                    res = pd.merge(selected[f"{data}_{ca_type}"]["filename"], res_df, on="filename", how="left")
                res["score"] = res.apply(
                    lambda x: cal_score(
                        x["labels_gt"], x["labels"], num_classes[data], int(ca_type.split("_")[-1]), cat2idxes[data]
                    ), axis=1
                )
                print(res)
                res.to_csv(os.path.join(dst_dir, f"res_{data}_{model}_{ca_type}_cmp_random_{i+1}.csv"), index=False)


if __name__ == "__main__":
    with open("errors.txt", 'w') as f:
        f.write("")
    for task in TASKS:
        print("task: {} started".format(task["task_name"]))
        start = time.time()
        args = argparse.Namespace(**task["args"])
        args.dataloader = task["dataloader"]
        try:
            runner(args)
        except Exception as e:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
    randomly_select(10, 5)
