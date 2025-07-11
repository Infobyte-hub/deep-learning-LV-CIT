import os
import pandas as pd
import glob
import warnings
import joblib


warnings.filterwarnings('ignore')

res_root = os.path.join("data", "lvcit", "4results", "atom")
models = [
    ("msrn", "voc"),
    ("mlgcn", "voc"),
    ("asl", "voc"),
    ("msrn", "coco"),
    ("mlgcn", "coco"),
    ("asl", "coco"),
]


def check(x, inter_set=False, union_set=False):
    title = x["title"]
    if inter_set:
        for pred in x.tolist()[1:]:
            if not title.issubset(pred):
                return False
        else:
            return True
    elif union_set:
        all_lc = set.union(*x.tolist()[1:])
        if title.issubset(all_lc):
            return True
        elif sum([1 if {label}.issubset(all_lc) else 0 for label in title]) > 0:
            return False, True
        else:
            return False, False
    else:
        for pred in x.tolist()[1:]:
            if title.issubset(pred):
                return True
        else:
            return False


def gen_pkl():
    data_from = "google"
    mr_name = ['scale', 'rotation', 'contrast', 'saturation', 'brightness', 'sharp', 'gaussian']
    mr_name_df = pd.DataFrame({"mr": mr_name}).sort_values("mr")
    mr_name_df["key"] = 0
    for model, data in models:
        for w in [2]:
            print(data, w, model)
            result_path = glob.glob(os.path.join(
                res_root,
                f"result_{data_from}_{data}{w}",
                f"result_{data_from}_{data}_{w}way_{model}*.xlsx"
            ))[0]
            followup_result_path = glob.glob(os.path.join(
                res_root,
                f"result_{data_from}_{data}_followup{w}",
                f"result_{data_from}_{data}_followup_{w}way_{model}_map*.xlsx"
            ))[0]
            result_df = pd.read_excel(result_path).fillna("")
            result_df["pred"] = result_df[result_df.columns.difference(["img"])].apply(
                lambda x: set(sorted(filter(None, x.tolist()))), axis=1
            )
            followup_df = pd.read_excel(followup_result_path).fillna("")

            result_df["key"] = 0
            result_df = pd.merge(result_df, mr_name_df, on="key")
            result_df["mr_img"] = result_df.apply(lambda x: x["img"].split(".")[0] + "_" + x["mr"], axis=1)
            followup_tmp = pd.DataFrame()
            followup_tmp["mr_img"] = followup_df.apply(lambda x: x["img"].split(".")[0], axis=1)
            followup_tmp["mr_pred"] = followup_df[followup_df.columns.difference(["img"])].apply(
                lambda x: set(sorted(filter(None, x.tolist()))), axis=1
            )
            result_df = pd.merge(result_df, followup_tmp, on="mr_img")
            result_df = result_df[["img", "mr", "pred", "mr_pred"]]
            result_df = result_df.set_index(["img", 'mr']).unstack(level=-1).reset_index()
            result_df.columns = ["_".join(x) for x in result_df.columns.ravel()]
            result_df = result_df[
                ["img_", "pred_brightness"] + [x for x in result_df.columns if x.startswith("mr_pred")]]
            result_df.rename(columns={
                "img_": "img",
                "pred_brightness": "pred"
            }, inplace=True)
            result_df["title"] = result_df["img"].apply(
                lambda x: set(x.split("_")[0].split("-"))
            )

            gt_path = os.path.join(res_root, "vote", f"{data}{w}ground_vote.xlsx")
            gt_df = pd.read_excel(gt_path).rename(columns={"Image": "img", "Ground Truth": "gt"})
            result_df["img"] = result_df["img"].apply(
                lambda x: x.split(".")[0]
            )
            result_df = pd.merge(result_df, gt_df, on="img", how="left")

            result_df["any_match"] = result_df[["title", "pred"] + ["mr_pred_" + mr for mr in mr_name]].apply(
                lambda x: check(x),
                axis=1
            )
            result_df["inter_match"] = result_df[["title", "pred"] + ["mr_pred_" + mr for mr in mr_name]].apply(
                lambda x: check(x, True, False),
                axis=1
            )
            result_df["union_match"] = result_df[["title", "pred"] + ["mr_pred_" + mr for mr in mr_name]].apply(
                lambda x: check(x, False, True),
                axis=1
            )
            result_df["mt"] = result_df[["pred"] + ["mr_pred_" + mr for mr in mr_name]].apply(
                lambda x: sum([1 if x["pred"] != mr else 0 for mr in x[1:]]),
                axis=1
            )
            result_df["lc"] = result_df[["title", "pred"]].apply(
                lambda x: 0 if x["title"].issubset(x["pred"]) else 1,
                axis=1
            )
            result_df["type"] = ""
            result_df.loc[(result_df["mt"] == 0) & (result_df["lc"] == 0), "type"] = "con"
            result_df.loc[(result_df["mt"] > 0) & (result_df["lc"] == 0), "type"] = "vul"
            result_df.loc[(result_df["mt"] == 0) & (result_df["lc"] > 0), "type"] = "inc"
            result_df.loc[(result_df["mt"] > 0) & (result_df["lc"] > 0), "type"] = "com"
            # print(result_df)
            joblib.dump(result_df, os.path.join(
                res_root,
                "all_info",
                f"{data_from}_{data}{w}_{model}.pkl"
            ))


if __name__ == "__main__":
    gen_pkl()
