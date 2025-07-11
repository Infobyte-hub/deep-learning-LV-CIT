import numpy as np
from itertools import combinations
import argparse
from decimal import Decimal


def myround(x, w):
    return float(Decimal(str(x)).quantize(Decimal("0." + "0" * (w - 1) + "1"), rounding="ROUND_HALF_UP"))


def cal_score(gt, pred, num_classes, way_num, cat2idx):
    gt_idx = np.zeros(num_classes, dtype=np.int32)
    pred_idx = np.zeros(num_classes, dtype=np.int32)
    gt_idx[[cat2idx[cat] for cat in filter(None, gt.split("|"))]] = 1
    pred_idx[[cat2idx[cat] for cat in filter(None, pred.split("|"))]] = 1
    gt_value = [f"{idx}_{val}" for idx, val in enumerate(gt_idx)]
    pred_value = [f"{idx}_{val}" for idx, val in enumerate(pred_idx)]
    # print(gt_value)
    # print(pred_value)
    gt_combinations = frozenset(combinations(gt_value, way_num))
    pred_combinations = frozenset(combinations(pred_value, way_num))
    # print(gt_combinations)
    # print(pred_combinations)
    pass_comb = gt_combinations.intersection(pred_combinations)
    # print(pass_comb)
    score = len(pass_comb) / len(gt_combinations)
    return score


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
