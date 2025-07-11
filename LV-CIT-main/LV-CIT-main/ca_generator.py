import math
import os
import random
import sys
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from itertools import combinations
from scipy.special import comb
import argparse
from util import str2bool


output_dir = os.path.join("data", "lvcit", "1covering_array")
np.random.seed(int(time.time()))
random.seed(int(time.time()))


class BitSet:
    def __init__(self, size, k, comb2idx, values=None):
        self.size = size
        self.k = k
        self.comb2idx = comb2idx
        self.data = bytearray(size // 8 + 1)
        if values is not None:
            self.init(values)

    def __setitem__(self, key, value):
        if key >= self.size:
            raise IndexError(f"index out of range: get index {key}, max index {self.size - 1}")
        if value == 1:
            self.data[key // 8] |= 1 << (key % 8)
        elif value == 0:
            self.data[key // 8] &= ~(1 << (key % 8))
        else:
            raise ValueError("value must be 0 or 1")

    def __getitem__(self, key):
        if key >= self.size:
            raise IndexError(f"index out of range: get index {key}, max index {self.size - 1}")
        return (self.data[key // 8] >> (key % 8)) & 1

    def __str__(self):
        return "".join([str(self[i]) for i in range(self.size)])

    def __len__(self):
        return sum([self[i] for i in range(self.size)])

    def copy(self):
        tmp = BitSet(self.size, self.k, self.comb2idx)
        tmp.update(self)
        return tmp

    def update(self, other):
        if self.size != other.size:
            raise ValueError("size must be equal")
        for i in range(self.size // 8 + 1):
            self.data[i] |= other.data[i]

    def clear(self):
        self.data = bytearray(self.size // 8 + 1)
        return self

    def init(self, values):
        for value in values:
            offset = 0
            for v in sorted(value):
                offset = offset * 2 + v[1]
            idx = self.comb2idx[tuple([v[0] for v in sorted(value)])] * pow(2, self.k) + offset
            self[idx] = 1

    @staticmethod
    def union(bitsets):
        tmp = bitsets[0].copy()
        for bitset in bitsets[1:]:
            tmp.update(bitset)
        return tmp


def calculate_coverage(array: DataFrame, label, tau, current_combinations=None, is_del=False, del_idx=0):
    """
    Calculate the k-dimensional coverage of the current array

    :param array: current covering array
    :param label: number of labels
    :param tau: covering strength
    :param current_combinations: the combinations that have been covered, type is dict if is_del is False else set
    :param is_del: calculate the coverage when removing rows from the array if true
    :param del_idx: the index of the row to be removed
    :return: coverage, number of tau-way combinations already covered, number of tau-way combinations, current_combinations
    """
    if is_del:
        del_key = None
        for key in current_combinations.keys():
            if key[0] <= del_idx < key[1]:
                del_key = key
                break
        current_combinations = {key: val if key != del_key else val.copy().clear() for key, val in current_combinations.items()}
        array.loc[(array.index >= del_key[0]) & (array.index < del_key[1]), "combinations"].apply(
            lambda x: current_combinations[del_key].update(x)
        )
        t_i = len(BitSet.union(list(current_combinations.values())))
    else:
        array["combinations"] = array[pd.isna(array["combinations"])].apply(
            lambda x: BitSet(
                current_combinations.size,
                current_combinations.k,
                current_combinations.comb2idx,
                combinations(enumerate(x[list(range(label))]), tau)
            ), axis=1
        )
        array["combinations"].apply(
            lambda x: current_combinations.update(x)
        )
        t_i = len(current_combinations)
    sut_i = int(comb(label, tau) * pow(2, tau))
    cov = t_i / sut_i
    return cov, t_i, sut_i, current_combinations


def add_lines_baseline(array: DataFrame, label, tau):
    """
    the Baseline method to generate covering array, C(n, tau)
    require label >= 2 * tau, otherwise invalid

    :param array: current covering array
    :param label: number of labels
    :param tau: covering strength
    :return: updated covering array, coverage
    """
    if label < 2 * tau:
        raise Exception(f"bad parameter, excepted label >= 2 * tau, get label: {label}, tau: {tau}")
    for combination in combinations(list(range(label)), tau):
        line = np.zeros(label, int)
        for index in combination:
            line[index] = 1
        array.loc[len(array.index), list(range(label))] = line
    print(f"\rsize: {len(array)}, coverage: 1", end="")
    return array, 1


def add_lines_adaptive_random(array: DataFrame, label, k, tau, current_combinations):
    """
    the adaptive random method in LV-CIT to generate covering array

    :param array: current covering array
    :param label: number of labels
    :param k: the counting constraint variable
    :param tau: covering strength
    :param current_combinations: the combinations that have been covered
    :return: updated covering array, coverage, and updated current_combinations
    """
    try:
        tmp = add_lines_adaptive_random.old_cov
        tmp = add_lines_adaptive_random.count
    except Exception:
        add_lines_adaptive_random.old_cov = 0
        add_lines_adaptive_random.count = 0

    # generate a line adaptive randomly
    def gen_lines(array):
        c = math.ceil(np.random.random(1) * k)
        label_count = array[list(range(label))].sum(axis=0)
        line = np.zeros(label, int)
        label_nsmaillest = [
            idx for idx, val in sorted(
                [(i, x) for i, x in enumerate(label_count)], key=lambda x: x[1]
            )[:int(np.random.random(1) * label * 0.5)]
        ]
        if add_lines_adaptive_random.count > 1000:
            label_nsmaillest = []
        others = list(set(range(label)).difference(set(label_nsmaillest)))
        if math.ceil(c/2) > len(label_nsmaillest):
            c1 = len(label_nsmaillest)
            c2 = c - c1
        elif c - math.ceil(c/2) > len(others):
            c2 = len(others)
            c1 = c - c2
        else:
            c1 = math.ceil(c/2)
            c2 = c - c1
        index_line = random.sample(label_nsmaillest, c1) + random.sample(others, c2)
        line[index_line] = 1
        return line

    # check if the new line contains uncovered combinations, if so, add it to the array
    tmp = pd.DataFrame(columns=array.columns)
    line = gen_lines(array)
    tmp.loc[len(tmp.index), list(range(label))] = line
    coverage0 = add_lines_adaptive_random.old_cov
    coverage1, t_i, sut_i, current_combinations_new = calculate_coverage(tmp, label, tau, current_combinations)
    if coverage0 < coverage1:
        array = pd.concat([array, tmp]).reset_index(drop=True)
        add_lines_adaptive_random.old_cov = coverage1
        print(
            f"\rsize: {len(array)}, coverage: {coverage1}, {t_i}/{sut_i}, count: {add_lines_adaptive_random.count}",
            end=""
        )
        add_lines_adaptive_random.count = 0
        return array, coverage1, current_combinations_new
    else:
        add_lines_adaptive_random.count += 1
        return array, coverage0, current_combinations


def del_lines(array: DataFrame, label, tau, thr):
    """
    delete rows from the covering array, if the coverage is still greater than the threshold after deletion

    :param array: covering array
    :param label: number of labels
    :param tau: covering strength
    :param thr: threshold
    :return: updated covering array
    """
    current_combinations = {}
    batch_size = int(math.sqrt(len(array)))
    for j in range(0, len(array), batch_size):
        array.loc[j: j + batch_size - 1, "combinations"].apply(
            lambda x: current_combinations[(j, j+batch_size)].update(x)
            if (j, j+batch_size) in current_combinations.keys() else current_combinations.update({(j, j+batch_size): x.copy()})
        )
    i = len(array) - 1
    while i >= 0:
        cov, _, _, current_combinations_tmp = calculate_coverage(
            array.drop(index=[i]), label, tau, current_combinations, True, i
        )
        if cov >= thr:
            array.drop(index=[i], inplace=True)
            current_combinations = current_combinations_tmp
            print(f"\rreduced size: {len(array)}", end="")
        i -= 1
    return array


def get_covering_array(method, label, k, tau, thr: float = 1.0):
    """
    Get the covering array by the specified method

    :param method: the method to generate covering array, baseline or adaptive random
    :param label: number of labels
    :param k: the maximum number of labels in a combination
    :param tau: the counting constraint variable
    :param thr: the threshold of coverage, default 100%
    :return: covering array, coverage
    """
    array = DataFrame(columns=list(range(label)) + ["combinations"])
    coverage = 0
    current_combinations = BitSet(
        int(comb(label, tau) * pow(2, tau)),
        tau,
        {cb: idx for idx, cb in enumerate(combinations(range(label), tau))}
    )
    add_lines_adaptive_random.old_cov = 0
    add_lines_adaptive_random.count = 0
    while coverage < thr:
        if method == "baseline":
            array, coverage = add_lines_baseline(array, label, tau)
        elif method == "adaptive random":
            array, coverage, current_combinations = add_lines_adaptive_random(array, label, k, tau, current_combinations)
    print("")
    array["c"] = array[list(range(label))].apply(lambda x: x.sum(), axis=True)
    array.sort_values(by=["c"] + list(range(label)), inplace=True, ascending=[True] + [False] * label)
    array.reset_index(drop=True, inplace=True)
    if method != "baseline":
        array = del_lines(array, label, tau, thr)
        print("")
    array = array[list(range(label)) + ["c"]]
    return array, coverage


def task(label, k, tau, method="random", thr=1.0):
    print(label, k, tau, method, thr)
    start = time.process_time()
    array, coverage = get_covering_array(method, label, k, tau, thr)
    end = time.process_time()
    # print(array)
    print(f"final size: {len(array)}, coverage: {coverage}, time: {end - start}")
    res_path = os.path.join(
        output_dir,
        method,
        f"ca_{method}_{label}_{k}_{tau}_{len(array)}_{end - start}.csv"
    )
    if not os.path.exists(os.path.dirname(res_path)):
        os.makedirs(os.path.dirname(res_path))
    array.to_csv(res_path, index=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 1000)

    parser = argparse.ArgumentParser(description="Generate covering array")
    parser.add_argument(
        "--method", "-m",
        type=str, default="adaptive random",
        help="method to generate covering array, baseline or adaptive random"
    )
    parser.add_argument("--all", "-a", type=str2bool, default=True, help="generate covering arrays for all")
    parser.add_argument("-n", type=int, default=20, help="number of labels (size of label space)")
    parser.add_argument("-k", type=int, default=4, help="the counting constraint value, default 4")
    parser.add_argument("-t", type=int, default=2, help="covering strength, default 2")
    parser.add_argument("--number", type=int, default=1, help="how many covering arrays to generate")
    args = parser.parse_args()
    if args.all:
        for n in [20, 80]:
            for k in range(2, 7):
                for _ in range(5):
                    tau = 2
                    task(n, k, tau, "baseline")
        for n in [20, 80]:
            for k in range(2, 7):
                for _ in range(5):
                    tau = 2
                    task(n, k, tau, "adaptive random")
    else:
        for _ in range(args.number):
            task(args.n, args.k, args.t, args.method)
