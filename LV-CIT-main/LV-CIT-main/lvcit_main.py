import traceback
import warnings
import time
import argparse
import os

from util import str2bool
from lvcit_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')

# data info
ROOT = os.path.join("data", "lvcit")
VERSION = "_v6_random_255_255_255_255_s1a0"

checkpoints_dir = "checkpoints"
checkpoints_save_dir = os.path.join("checkpoints", "save")
data_info = [
    {
        "data_name": "voc",
        "data": os.path.join(ROOT, "3composite_img", f"VOC_20{VERSION}"),
        "covering_array_type": [
            f"adaptive random_20_{k}_{tau}" for k in [4] for tau in [2]
        ],
        "num_classes": 20,
        "phase": "predict",
        "res_path": os.path.join(ROOT, "4results", f"VOC_20{VERSION}"),
        "inp_name": "data/voc/voc_glove_word2vec.pkl",
        "graph_file": "data/voc/voc_adj.pkl",
    },
    {
        "data_name": "coco",
        "data": os.path.join(ROOT, "3composite_img", f"COCO_80{VERSION}"),
        "covering_array_type": [
            f"adaptive random_80_{k}_{tau}" for k in [4] for tau in [2]
        ],
        "num_classes": 80,
        "phase": "predict",
        "res_path": os.path.join(ROOT, "4results", f"COCO_80{VERSION}"),
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
        "batch_size": 64,
        "threshold": 0.5,
        "workers": 1,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": os.path.join(checkpoints_dir, "msrn", "voc_checkpoint.pth.tar"),
        "evaluate": True,
        "pretrained": 1,
        "pretrain_model": os.path.join(checkpoints_dir, "msrn", "resnet101_for_msrn.pth.tar"),
        "pool_ratio": 0.2,
        "backbone": "resnet101",
        "save_model_path": os.path.join(checkpoints_save_dir, "msrn"),
    },
    # 1 ml gcn
    {
        "model_name": "mlgcn",
        "image_size": 448,
        "batch_size": 64,
        "threshold": 0.5,
        "workers": 2,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": os.path.join(checkpoints_dir, "mlgcn", "voc_checkpoint.pth.tar"),
        "evaluate": True,
        "save_model_path": os.path.join(checkpoints_save_dir, "mlgcn"),
    },
    # 2 asl
    {
        "model_name": "asl",
        "model_type": "tresnet_xl",
        "model_path": os.path.join(checkpoints_dir, "asl", "voc_checkpoint.pth"),
        "workers": 1,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 64,
        "print_freq": 64,
        # TODO save path
    },
]

TASKS = [
    # voc msrn
    {
        "task_name": "voc_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader": LvcitVoc,
    },
    # voc mlgcn
    {
        "task_name": "voc_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader": LvcitVoc,
    },
    # voc asl
    {
        "task_name": "voc_asl",
        "args": {**data_info[0], **model_info[2]},
        "dataloader": LvcitVoc2,
    },

    # coco msrn
    {
        "task_name": "coco_msrn",
        "args": {
            **data_info[1], **model_info[0],
            "pool_ratio": 0.05,
            "resume": os.path.join(checkpoints_dir, "msrn", "coco_checkpoint.pth.tar"),
            "batch_size": 10,
        },
        "dataloader": LvcitCoco,
    },
    # coco mlgcn
    {
        "task_name": "coco_mlgcn",
        "args": {
            **data_info[1], **model_info[1],
            "resume": os.path.join(checkpoints_dir, "mlgcn", "coco_checkpoint.pth.tar"),
            "batch_size": 80,
        },
        "dataloader": LvcitCoco,
    },
    # coco asl
    {
        "task_name": "coco_asl",
        "args": {
            **data_info[1], **model_info[2],
            "model_type": "tresnet_l",
            "model_path": os.path.join(checkpoints_dir, "asl", "coco_checkpoint.pth"),
            "print_freq": 640,
            "batch_size": 64,
        },
        "dataloader": LvcitCoco2,
    },
]


if __name__ == "__main__":
    with open("errors.txt", 'w') as f:
        f.write("")
    parser = argparse.ArgumentParser(description="test execution")
    parser.add_argument(
        "--demo", "-d",
        type=str2bool,
        default=False,
    )
    args = parser.parse_args()
    if args.demo:
        task = TASKS[0]
        task["args"]["covering_array_type"] = ["adaptive random_6_3_2"]
        print("task: {} started".format(task["task_name"]))
        start = time.time()
        args = argparse.Namespace(**task["args"])
        args.dataloader = task["dataloader"]
        args.data = os.path.join(args.data, args.model_name)
        args.res_path = os.path.join(args.res_path, args.model_name)
        try:
            runner(args, 1)
        except Exception:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
    else:
        for task in TASKS:
            print("task: {} started".format(task["task_name"]))
            start = time.time()
            args = argparse.Namespace(**task["args"])
            args.dataloader = task["dataloader"]
            args.data = os.path.join(args.data, args.model_name)
            args.res_path = os.path.join(args.res_path, args.model_name)
            try:
                runner(args)
            except Exception:
                with open("errors.txt", 'a') as f:
                    f.write(task["task_name"])
                    traceback.print_exc()
                    f.write(traceback.format_exc())
                    f.write("\n")
            print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
