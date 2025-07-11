import glob
import os
import time
import joblib
import pandas as pd
import itertools
from ca_generator import calculate_coverage, BitSet
from scipy.special import comb
from itertools import combinations
from functools import reduce
from util import myround
import shutil
import json


result_root = os.path.join("data", "lvcit", "4results")
anl_result_root = os.path.join("data", "lvcit", "5res_analyse")
models = [
    ("msrn", "voc"),
    ("mlgcn", "voc"),
    ("asl", "voc"),
    ("msrn", "coco"),
    ("mlgcn", "coco"),
    ("asl", "coco"),
]
ca_types = {
    "voc": [
        f"adaptive random_20_{k}_{tau}" for tau in [2] for k in [4]
    ],
    "coco": [
        f"adaptive random_80_{k}_{tau}" for tau in [2] for k in [4]
    ],
}
res_dir = {
    "voc": "VOC_20_v6_random_255_255_255_255_s1a0",
    "coco": "COCO_80_v6_random_255_255_255_255_s1a0",
}
labels = {
    "voc": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant",
            "sheep", "sofa", "train", "tv"],
    "coco": ["airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove",
             "bear", "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl",
             "broccoli", "bus", "cake", "car", "carrot", "cat", "cell phone", "chair", "clock",
             "couch", "cow", "cup", "dining table", "dog", "donut", "elephant",
             "fire hydrant", "fork", "frisbee", "giraffe", "hair dryer", "handbag", "horse",
             "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle",
             "mouse", "orange", "oven", "parking meter", "person", "pizza", "potted plant",
             "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink", "skateboard",
             "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
             "surfboard", "teddy bear", "tennis racket", "tie", "toaster", "toilet",
             "toothbrush", "traffic light", "train", "truck", "tv", "umbrella", "vase",
             "wine glass", "zebra"],
}


def cal_lc_coverage(pred_df, label_num, cover_k):
    current_combinations = BitSet(
        int(comb(label_num, cover_k) * pow(2, cover_k)),
        cover_k,
        {cb: idx for idx, cb in enumerate(combinations(range(label_num), cover_k))}
    )
    tmp_df = pred_df.copy()
    tmp_df["combinations"] = None
    coverage, cover_lc, all_lc, _ = calculate_coverage(tmp_df, label_num, cover_k, current_combinations)
    return coverage, cover_lc, all_lc


def cal_error(pred_df, gt_df, cover_k, label):
    def check_error(x, cover_k):
        pred_labels = [f"{l}_{v}" for l, v in zip(label, x[[f"{l}_pred" for l in label]])]
        gt_labels = [f"{l}_{v}" for l, v in zip(label, x[[f"{l}_gt" for l in label]])]
        pred_lcs = set(combinations(pred_labels, cover_k))
        gt_lcs = set(combinations(gt_labels, cover_k))
        return gt_lcs.difference(pred_lcs)

    # print(pred_df)
    tmp_df = pd.concat(
        [
            pred_df.rename(columns={l: l + "_pred" for l in pred_df.columns}, inplace=False),
            gt_df.rename(columns={l: l + "_gt" for l in gt_df.columns}, inplace=False)
        ],
        axis=1, join="outer"
    )
    tmp_df["error"] = tmp_df.apply(lambda x: check_error(x, cover_k), axis=1)
    error_num = tmp_df["error"].apply(lambda x: 1 if len(x) else 0).sum()
    error_type_num = len(tmp_df["error"].agg(lambda x: set.union(*x)))
    return error_num, error_type_num


def calculate():
    metrics_df1 = pd.DataFrame(columns=[
        "way", "data", "model", "k",
        "img_num", "random_img_num",
        "input_coverage", "random_input_coverage",
        "output_coverage", "random_output_coverage",
        "error_num", "random_error_num",
        "error_type_num", "random_error_type_num",
        "% of error", "% of random_error",
        "error_diversity", "random_error_diversity"
    ])
    metrics_df2 = pd.DataFrame(columns=[
        "way", "data", "model", "k", "score"
    ])

    def cal_once(res_path):
        res_df = pd.read_csv(res_path).fillna('')
        pred_df = res_df[["labels"]].apply(
            lambda x: [1 if lab in x[0].split("|") else 0 for lab in labels[data]], axis=1, result_type="expand"
        )
        gt_df = res_df[["labels_gt"]].apply(
            lambda x: [1 if lab in x[0].split("|") else 0 for lab in labels[data]], axis=1, result_type="expand"
        )
        # counting_df = pd.DataFrame()
        # counting_df["count"] = gt_df.drop_duplicates().apply(lambda x: sum(x), axis=1)
        # counting_df = counting_df.groupby("count").size().reset_index(name="num")
        # counting_df.loc[len(counting_df)] = ["mean", (counting_df["num"] * counting_df["count"]).sum() / counting_df["num"].sum()]
        # print(counting_df)
        pred_df.columns = labels[data]
        gt_df.columns = labels[data]

        # input_coverage
        input_coverage, _, _ = cal_lc_coverage(
            gt_df, len(labels[data]), way
        )
        # output_coverage
        output_coverage, _, _ = cal_lc_coverage(
            pred_df, len(labels[data]), way
        )
        error_num, error_type_num = cal_error(pred_df, gt_df, way, labels[data])

        # mIA
        mia = myround(res_df["score"].mean(), 4)
        # return 1, 0, 0, 0, 0, mia
        return len(res_df), input_coverage, output_coverage, error_num, error_type_num, mia

    print(metrics_df1.columns.tolist())
    if os.path.exists(os.path.join(anl_result_root, "metrics_df1.pkl")):
        metrics_df1 = pd.read_pickle(os.path.join(anl_result_root, "metrics_df1.pkl"))
        metrics_df2 = pd.read_pickle(os.path.join(anl_result_root, "metrics_df2.pkl"))
    for model, data in models:
        for ca_type in ca_types[data]:
            way = int(ca_type.split("_")[-1])
            k = int(ca_type.split("_")[-2])
            # print(data, model, way, ca_type)
            line = [way, data.upper(), model.upper(), k]
            if len(metrics_df1[(metrics_df1["way"] == way) & (metrics_df1["data"] == data.upper()) & (metrics_df1["model"] == model.upper()) & (metrics_df1["k"] == k)]):
                print(metrics_df1[(metrics_df1["way"] == way) & (metrics_df1["data"] == data.upper()) & (metrics_df1["model"] == model.upper()) & (metrics_df1["k"] == k)].values.tolist())
                continue
            img_num, input_coverage, output_coverage, error_num, error_type_num, mia = [], [], [], [], [], []
            for i in range(5):
                res_path = os.path.join(
                    result_root,
                    res_dir[data],
                    model,
                    f"{ca_type}_No{i+1}",
                    f"res_{data}_{model}_{ca_type}_No{i+1}.csv"
                )
                (
                    img_num_i,
                    input_coverage_i,
                    output_coverage_i,
                    error_num_i,
                    error_type_num_i,
                    mia_i
                ) = cal_once(res_path)
                img_num.append(img_num_i)
                input_coverage.append(input_coverage_i)
                output_coverage.append(output_coverage_i)
                error_num.append(error_num_i)
                error_type_num.append(error_type_num_i)
                mia.append(mia_i)

            metrics_df2.loc[len(metrics_df2)] = line + ["{:.2f}".format(
                myround(sum(mia) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%"]

            random_img_num, random_input_coverage, random_output_coverage, random_error_num, random_error_type_num = [], [], [], [], []
            for i in range(5):
                random_res_path = os.path.join(
                    result_root,
                    res_dir[data],
                    "random",
                    f"{ca_type}_No{i+1}",
                    f"res_{data}_{model}_{ca_type}_cmp_random_{i+1}.csv"
                )
                (
                    random_img_num_i,
                    random_input_coverage_i,
                    random_output_coverage_i,
                    random_error_num_i,
                    random_error_type_num_i,
                    _
                ) = cal_once(random_res_path)

                random_img_num.append(random_img_num_i)
                random_input_coverage.append(random_input_coverage_i)
                random_output_coverage.append(random_output_coverage_i)
                random_error_num.append(random_error_num_i)
                random_error_type_num.append(random_error_type_num_i)

            # img num
            line.append("{:.2f}".format(
                myround(sum(img_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            line.append("{:.2f}".format(
                myround(sum(random_img_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            # input_coverage
            line.append("{:.2f}".format(
                myround(sum(input_coverage) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            line.append("{:.2f}".format(
                myround(sum(random_input_coverage) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            # output_coverage
            line.append("{:.2f}".format(
                myround(sum(output_coverage) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            line.append("{:.2f}".format(
                myround(sum(random_output_coverage) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            # error_num
            line.append("{:.2f}".format(
                myround(sum(error_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            line.append("{:.2f}".format(
                myround(sum(random_error_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            # error_type_num
            line.append("{:.2f}".format(
                myround(sum(error_type_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            line.append("{:.2f}".format(
                myround(sum(random_error_type_num) / 5, 2)
            ).rstrip("0").rstrip("."))
            # error percent
            line.append("{:.2f}".format(
                myround(sum([e / l for e, l in zip(error_num, img_num)]) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            line.append("{:.2f}".format(
                myround(sum([e / l for e, l in zip(random_error_num, random_img_num)]) / 5 * 100, 2)
            ).rstrip("0").rstrip(".") + "%")
            # error type per img
            line.append("{:.2f}".format(
                myround(sum([e / l for e, l in zip(error_type_num, img_num)]) / 5, 2)
            ).rstrip("0").rstrip("."))
            line.append("{:.2f}".format(
                myround(sum([e / l for e, l in zip(random_error_type_num, random_img_num)]) / 5, 2)
            ).rstrip("0").rstrip("."))
            print(line)
            metrics_df1.loc[len(metrics_df1)] = line

            # train_df = pd.read_csv(os.path.join(
            #     anl_result_root, "train_val_data", f"{data}_train_anno.csv"
            # ))
            # _, final_score_df = cal_lc_score(pred_df, gt_df, way, train_df)
            # print(final_score_df)
            # final_score_df.to_excel(os.path.join(
            #     anl_result_root, "lcs_score", f"lcs_scores_{data}_{model}_{way}.xlsx"
            # ), index=False)

            metrics_df1.to_pickle(os.path.join(anl_result_root, "metrics_df1.pkl"))
            metrics_df2.to_pickle(os.path.join(anl_result_root, "metrics_df2.pkl"))

    print(metrics_df1)
    print(metrics_df2)
    return metrics_df1, metrics_df2
    # return None, None


def classify_errors():
    for model, data in models:
        train_df = pd.read_csv(os.path.join(
            anl_result_root, "train_val_data", f"{data}_train_anno.csv"
        ))
        lc_num_of_train = {}
        if not os.path.exists(os.path.join(anl_result_root, f"lc_num_of_train_{data}.pkl")):
            for way in [1, 2]:
                for label_comb in itertools.combinations(labels[data], way):
                    lc_num_of_train["|".join(label_comb)] = train_df[train_df.apply(
                        lambda x: all([x[lab] for lab in label_comb]), axis=1
                    )].shape[0]
            with open(os.path.join(anl_result_root, f"lc_num_of_train_{data}.pkl"), "wb") as f:
                joblib.dump(lc_num_of_train, f)
        else:
            with open(os.path.join(anl_result_root, f"lc_num_of_train_{data}.pkl"), "rb") as f:
                lc_num_of_train = joblib.load(f)

        for ca_type in ca_types[data]:
            error_type_df = pd.DataFrame(columns=[
                "label_comb",
                "pass", "pass_ratio",
                "missing", "missing_ratio",
                "extra", "extra_ratio",
                "switch", "switch_ratio",  # num of images of four types
                "probably",  # the error type the lc probably belongs to
                "num_in_trainset",  # num of images in train set
                "missing_images",  # images of this type
                "extra_images",
                "switch_images",  # mismatch images
            ])
            way = int(ca_type.split("_")[-1])
            k = int(ca_type.split("_")[-2])
            print(data, model, way, ca_type)
            error_type_path = os.path.join(
                anl_result_root,
                "error_type",
                f"error_type_{data}_{model}_{k}_{way}.csv"
            )
            # if os.path.exists(error_type_path):
            #     continue
            for i in range(5):
                res_path = os.path.join(
                    result_root,
                    res_dir[data],
                    model,
                    f"{ca_type}_No{i+1}",
                    f"res_{data}_{model}_{ca_type}_No{i+1}.csv"
                )
                res_df = pd.read_csv(res_path).fillna('')
                res_df["tmp"] = res_df["filename"] + ":" + res_df["labels_gt"] + "->" + res_df["labels"]
                pred_df = res_df[["labels"]].apply(
                    lambda x: [1 if lab in x[0].split("|") else 0 for lab in labels[data]], axis=1, result_type="expand"
                )
                gt_df = res_df[["labels_gt"]].apply(
                    lambda x: [1 if lab in x[0].split("|") else 0 for lab in labels[data]], axis=1, result_type="expand"
                )
                pred_df.columns = labels[data]
                gt_df.columns = labels[data]
                img_num = len(res_df)
                for label_comb in itertools.combinations(labels[data], way):
                    tmp_pred_df = pred_df[list(label_comb)]
                    tmp_gt_df = gt_df[list(label_comb)]
                    differ = tmp_pred_df - tmp_gt_df
                    differ["type"] = differ.apply(
                        lambda x:
                        "pass" if all([d == 0 for d in x]) else
                        "extra" if all([d >= 0 for d in x]) else
                        "missing" if all([d <= 0 for d in x]) else
                        "switch",
                        axis=1,
                    )
                    differ.loc[(tmp_pred_df.apply(
                        lambda x: not any(x), axis=1
                    )) & (tmp_gt_df.apply(
                        lambda x: not all(x), axis=1
                    )), "type"] = "None"
                    differ.loc[tmp_gt_df.apply(
                        lambda x: not any(x), axis=1
                    ), "type"] = "None"
                    line = [
                        "|".join(label_comb),
                        differ[differ["type"] == "pass"].shape[0],
                        differ[differ["type"] == "pass"].shape[0] / img_num,
                        differ[differ["type"] == "missing"].shape[0],
                        differ[differ["type"] == "missing"].shape[0] / img_num,
                        differ[differ["type"] == "extra"].shape[0],
                        differ[differ["type"] == "extra"].shape[0] / img_num,
                        differ[differ["type"] == "switch"].shape[0],
                        differ[differ["type"] == "switch"].shape[0] / img_num,
                        differ[(differ["type"] != "pass") & (differ["type"] != "None")]["type"].value_counts().idxmax()
                        if differ[(differ["type"] != "pass") & (differ["type"] != "None")].shape[0] else "pass",
                        lc_num_of_train["|".join(label_comb)],
                        set(res_df[differ["type"] == "missing"]["tmp"]),
                        set(res_df[differ["type"] == "extra"]["tmp"]),
                        set(res_df[differ["type"] == "switch"]["tmp"]),
                    ]
                    error_type_df.loc[len(error_type_df)] = line
            aggs = {name: "mean" for name in error_type_df.columns[1:9]}
            aggs.update({
                "probably": "first",
                "num_in_trainset": "first",
                "missing_images": lambda sets: reduce(lambda x, y: x.union(y), sets),
                "extra_images": lambda sets: reduce(lambda x, y: x.union(y), sets),
                "switch_images": lambda sets: reduce(lambda x, y: x.union(y), sets),
            })
            error_type_df = error_type_df.groupby("label_comb").agg(aggs).reset_index()
            for i in range(way):
                error_type_df[f"label_{i+1}_in_trainset"] = error_type_df["label_comb"].apply(
                    lambda x: lc_num_of_train[sorted(x.split("|"))[i]]
                ) - error_type_df["num_in_trainset"]
            error_type_df.to_csv(os.path.join(error_type_path))


def cal_atom_mr():
    metrics_atom = pd.DataFrame(columns=[
        "model", "dataset", "way",
        "# of images",
        "# of errors",
        "% of errors",
    ])
    for way in [1, 2]:
        for model, dataset in models:
            info_df = joblib.load(os.path.join(
                result_root,
                "google",
                "all_info",
                f"google_{dataset}{way}_{model}.pkl"
            ))
            final_df = info_df
            error_num = len(final_df) - final_df[["mt"]].groupby("mt").size()[0]

            line = [
                model, dataset, way,
                len(final_df),
                error_num,
                "{:.2f}".format(myround(error_num / len(final_df) * 100, 2)).rstrip("0").rstrip(".") + "%"
            ]
            print(line)
            metrics_atom.loc[len(metrics_atom)] = line
    return metrics_atom


if __name__ == "__main__":
    if not os.path.exists(anl_result_root):
        os.makedirs(anl_result_root)

    metrics_df1, metrics_df2 = calculate()
    metrics_df1.to_excel(os.path.join(anl_result_root, "metrics_errs.xlsx"), index=False)
    metrics_df2.to_excel(os.path.join(anl_result_root, "metrics_mias.xlsx"), index=False)

    metrics_atom = cal_atom_mr()
    metrics_atom.to_excel(os.path.join(anl_result_root, "metrics_atom_mr.xlsx"), index=False)

    classify_errors()
