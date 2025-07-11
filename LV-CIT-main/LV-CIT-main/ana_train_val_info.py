import os
import pandas as pd
from tqdm import tqdm
import torch
from dataloaders.default_voc2 import Voc2007Classification2
from dataloaders.default_coco2 import COCO2014Classification2
from util import myround

data_root = "data"
res_root = os.path.join("data", "lvcit", "5res_analyse", "train_val_data")


def gen_train_val_data():
    voc_inp = os.path.join("data", "voc", "voc_glove_word2vec.pkl")
    coco_inp = os.path.join("data", "coco", "coco_glove_word2vec.pkl")
    datasets = []
    train_dataset_voc = Voc2007Classification2(os.path.join(data_root, "voc"), "trainval", inp_name=voc_inp)
    train_dataset_coco = COCO2014Classification2(os.path.join(data_root, "coco", "coco"), phase="train", inp_name=coco_inp)
    val_dataset_voc = Voc2007Classification2(os.path.join(data_root, "voc"), phase="test", inp_name=voc_inp)
    val_dataset_coco = COCO2014Classification2(os.path.join(data_root, "coco", "coco"), phase="val", inp_name=coco_inp)
    datasets.append(("train", "voc", train_dataset_voc))
    datasets.append(("train", "coco", train_dataset_coco))
    datasets.append(("val", "voc", val_dataset_voc))
    datasets.append(("val", "coco", val_dataset_coco))
    for phase, data, dataloader in datasets:
        line = [phase, data, 0, len(dataloader)]
        print(f"processing {data} {phase} data")
        cat2id = dataloader.get_cat2id()
        id2cat = list(cat2id.keys())
        df = pd.DataFrame(columns=["img"] + id2cat)
        for filename, _, target in tqdm(dataloader):
            if isinstance(target, torch.Tensor):
                target = target.to(torch.int).tolist()
            else:
                target = target.astype(int).tolist()
            # print(target)
            df.loc[len(df)] = [filename] + target
        df.to_csv(os.path.join(res_root, f"{data}_{phase}_anno.csv"), index=False)


def count_label_num():
    for data in ["voc", "coco"]:
        for phase in ["train", "val"]:
            print(f"processing {data} {phase} data")
            df = pd.read_csv(os.path.join(res_root, f"{data}_{phase}_anno.csv"))
            df["num"] = df[df.columns.difference(["img"])].apply(lambda x: len(x[x > 0]), axis=1)
            df = df[["num"]].groupby("num").size().reset_index(name="count")
            # df.to_csv(os.path.join(res_root, f"{data}_{phase}_num.csv"), index=False)
            total = df["count"].sum()

            df1 = df.iloc[list(range(6)) if len(df) >= 6 else list(range(len(df)))]
            df2 = df.iloc[df.index.difference(df1.index)]
            df = df1.append(pd.DataFrame([{"num": "6+", "count": df2["count"].sum()}]))
            df["percent"] = df["count"].apply(lambda x: "{:.2f}".format(myround(x / total * 100, 2)).rstrip("0").rstrip(".") + "%")
            print(df)
            df.to_csv(os.path.join(res_root, f"{data}_{phase}_num.csv"), index=False)


if __name__ == "__main__":
    gen_train_val_data()
    count_label_num()
