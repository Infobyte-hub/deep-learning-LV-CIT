import csv
import random
import time
import imutils
import numpy as np
import pandas as pd
import cv2
import math
import os
from shapely import affinity, geometry
import glob
from itertools import combinations
from tqdm import tqdm
import warnings
from collections import defaultdict
import threading
import argparse
from util import str2bool
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join("data", "lvcit")
DATA_NAMES = ["VOC", "COCO"]
MATTING_IMG_DIRS = {
    "VOC": os.path.join(DATA_DIR, "2matting_img", "VOC_library"),
    "COCO": os.path.join(DATA_DIR, "2matting_img", "COCO_library"),
}
COMPOSITE_IMG_DIRS = {
    "VOC": os.path.join(DATA_DIR, "3composite_img", "VOC_20"),
    "COCO": os.path.join(DATA_DIR, "3composite_img", "COCO_80"),
}
MODELS = {
    "VOC": ["msrn", "mlgcn", "asl"],
    "COCO": ["msrn", "mlgcn", "asl"],
}
COVERING_ARRAY_DIR = os.path.join(DATA_DIR, "1covering_array")
MAX_ANGLE = 10
DO_COMPONENTS = False
CA_METHOD = "adaptive random"

# version
DO_SCALE = True
DO_ANGLE = False
BACKGROUND_COLOR = (255, 255, 255, 255)
COMPOSITE_METHOD = "random"
VERSION = f"v6_{COMPOSITE_METHOD}_{'_'.join([str(i) for i in BACKGROUND_COLOR])}_s{1 if DO_SCALE else 0}a{1 if DO_ANGLE else 0}"


def get_matting_img(input_dir, model_name=None):
    matting_img = pd.DataFrame(columns=["labelid", "label", "file"])
    if model_name:
        object_detect_df = pd.read_csv(os.path.join(input_dir, "object_detect.csv"))
        count = object_detect_df["target"].drop_duplicates().count()
        cat2id = {cat: id for id, cat in enumerate(sorted(object_detect_df["target"].drop_duplicates()))}
        matting_img[["labelid", "label", "file"]] = object_detect_df[
            object_detect_df[model_name] == 1
        ].apply(
            lambda x: (cat2id[x["target"]], x["target"], os.path.join(input_dir, x["target"], x["filename"])),
            axis=1,
            result_type="expand",
        )
    else:
        fileinfo = os.walk(input_dir)
        count = 0
        for label in fileinfo.__next__()[1]:
            label_dir = os.path.join(input_dir, label)
            for path, dir_list, file_list in os.walk(label_dir):
                for file_name in file_list:
                    file = os.path.join(path, file_name)
                    line = [count, label, file]
                    matting_img.loc[len(matting_img.index)] = line
            count += 1
    matting_img["count"] = matting_img.apply(lambda x: 0, axis=1)
    return matting_img, count


def select(matting_img: pd.DataFrame, line, num, max_time, method):
    """
    select object images for test cases randomly

    :param matting_img: object images
    :param line: a line in the covering array
    :param num: number of test images for each test case
    :param max_time: maximum number of times an object image can be selected
    :param method: random or order
    :return: selected object images
    """
    labels = [index for (index, value) in enumerate(line) if value == 1]
    imgs = []
    imgs_set = []
    max_retry_time = 100
    while num > 0:
        imgs_i = []
        imgs_tmp = []
        for label in labels:
            if len(matting_img.loc[
                matting_img["labelid"] == label
            ]) == 0:
                continue
            library = matting_img.loc[
                (matting_img["labelid"] == label) & ((max_time == 0) | (matting_img["count"] < max_time)),
                ["file", "label"]
            ]
            while len(library) == 0:
                max_time += 1
                # print("\033[31m" + f"\rWarning: max_times increase to {max_time} due to label {label}" + "\033[0m", end="")
                library = matting_img.loc[
                    (matting_img["labelid"] == label) & (matting_img["count"] < max_time),
                    ["file", "label"]
                ]
            if method == "random":
                index = random.randint(0, len(library) - 1)
                x = library.iloc[index:index + 1]
            elif method == "order":
                x = library.iloc[len(imgs) % len(library):len(imgs) % len(library) + 1]
            else:
                raise Exception("method error")
            imgs_i.append((x.iloc[0]["file"], x.iloc[0]["label"]))
            imgs_tmp.append(x)
        if set(imgs_i) not in imgs_set:
            for x in imgs_tmp:
                matting_img.loc[x.index, "count"] += 1
            imgs.append(imgs_i)
            imgs_set.append(set(imgs_i))
            num -= 1
        else:
            if max_retry_time == 0:
                # print("\033[31m" + f"\rWarning: max_retry_time reached for {labels}, final size: {len(imgs)}" + "\033[0m", end="")
                break
            max_retry_time -= 1
    return imgs


def scale(img, scale_range):
    """
    scale the image randomly

    :param img: image
    :param scale_range: scale range, sr in [sr[0], sr[1])
    :return: scaled image
    """
    scale_min = scale_range[0]
    scale_max = scale_range[1]
    s = random.random() * (scale_max - scale_min) + scale_min
    img = cv2.resize(img, None, fx=s, fy=s)
    # cv2.imshow(f"scale-{s}", img)
    # cv2.waitKey(0)
    return img


def angle(img):
    """
    rotate the image randomly

    :param img: image
    :return: rotated image
    """
    alpha = random.random() * MAX_ANGLE * 2 - MAX_ANGLE  # [-MAX_ANGLE°, MAX_ANGLE°)
    # alpha = 0
    w = img.shape[0]
    h = img.shape[1]
    img = imutils.rotate_bound(img, alpha)
    w_ = img.shape[0]
    h_ = img.shape[1]
    w_cosa = w * abs(math.cos(alpha/180*math.pi))
    w_d_w_cosa = w_ - w_cosa
    h_cosa = h * abs(math.cos(alpha/180*math.pi))
    h_d_h_cosa = h_ - h_cosa
    if alpha < 90 or 180 < alpha < 270:
        poly = [
            np.array([[0, 0], [0, w_cosa+1], [h_d_h_cosa+1, 0]], np.int32),
            np.array([[0, w_], [h_cosa+1, w_], [0, w_cosa-1]], np.int32),
            np.array([[h_, w_], [h_, w_d_w_cosa-1], [h_cosa-1, w_]], np.int32),
            np.array([[h_, 0], [h_d_h_cosa-1, 0], [h_, w_d_w_cosa+1]], np.int32),
        ]
        vertex = [[w_cosa, 0], [w_, h_cosa], [w_d_w_cosa, h_], [0, h_d_h_cosa]]
    else:
        poly = [
            np.array([[0, 0], [0, w_d_w_cosa+1],  [h_cosa+1, 0]], np.int32),
            np.array([[0, w_], [h_d_h_cosa+1, w_], [0, w_d_w_cosa-1]], np.int32),
            np.array([[h_, w_], [h_, w_cosa-1], [h_d_h_cosa, w_-1]], np.int32),
            np.array([[h_, 0], [h_cosa-1, 0], [h_, w_cosa+1]], np.int32),
        ]
        vertex = [[w_d_w_cosa, 0], [w_, h_d_h_cosa], [w_cosa, h_], [0, h_cosa]]
    cv2.fillPoly(img, poly, (0, 0, 0, 0))
    return img, vertex


def get_center_random(polys, poly, shape, overlap_range):
    """
    get the center of the image randomly

    :param polys: all polygons
    :param poly: current polygon
    :param shape: shape of the image
    :param overlap_range: overlap range, or in [or[0], or[1])
    :return: center of the image
    """
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = affinity.translate(poly, -shape[0] / 2, -shape[1] / 2)

    beta = random.random() * 360
    overlap = random.random() * (overlap_range[1] - overlap_range[0]) + overlap_range[0]
    rho_min = 0
    rho_max = 1000
    while rho_max - rho_min > 1:
        center_min = [rho_min * math.cos(beta), rho_min * math.sin(beta)]
        center_max = [rho_max * math.cos(beta), rho_max * math.sin(beta)]
        poly_min = affinity.translate(poly, center_min[0], center_min[1])
        poly_max = affinity.translate(poly, center_max[0], center_max[1])
        intersects_min = polys.intersects(poly_min)
        intersects_max = polys.intersects(poly_max)
        if intersects_min and intersects_max:
            rho_min, rho_max = rho_max, rho_max * 2 - rho_min
        elif intersects_min and not intersects_max:
            rho_min, rho_max = (rho_min + rho_max) / 2, rho_max
        elif not intersects_min and intersects_max:
            rho_min, rho_max = rho_min, (rho_min + rho_max) / 2
        elif not intersects_min and not intersects_max:
            rho_min, rho_max = max(rho_min * 2 - rho_max, 0), rho_min
        # print(rho_min, rho_max)
    rho_max = rho_max * (1 - overlap)
    center = [int(rho_max * math.cos(beta)), int(rho_max * math.sin(beta))]
    # print(f"center: {center}")

    return center, affinity.translate(poly, center[0], center[1])


def get_background(size_x, size_y):
    """
    get the background image with size_x * size_y
    """
    background_img = np.zeros((size_x, size_y, 4), np.uint8)
    for i in range(4):
        background_img[:, :, i].fill(BACKGROUND_COLOR[i])
    return background_img


def paste_img(src, dst, center, translation_x, translation_y):
    x1 = center[0] - math.floor(src.shape[0] / 2) - translation_x
    x2 = center[0] + math.ceil(src.shape[0] / 2) - translation_x
    y1 = center[1] - math.floor(src.shape[1] / 2) - translation_y
    y2 = center[1] + math.ceil(src.shape[1] / 2) - translation_y
    b_channel, g_channel, r_channel, a_channel = cv2.split(src)
    b_channel2, g_channel2, r_channel2, a_channel2 = cv2.split(dst[x1:x2, y1:y2])
    dst[x1:x2, y1:y2] = cv2.merge([
        np.bitwise_or(np.bitwise_and(b_channel, a_channel), np.bitwise_and(b_channel2, np.bitwise_not(a_channel))),
        np.bitwise_or(np.bitwise_and(g_channel, a_channel), np.bitwise_and(g_channel2, np.bitwise_not(a_channel))),
        np.bitwise_or(np.bitwise_and(r_channel, a_channel), np.bitwise_and(r_channel2, np.bitwise_not(a_channel))),
        np.bitwise_or(a_channel, a_channel2)
    ])


def compare(img1, img2):
    if img1 is None or img2 is None:
        raise Exception(f"{'img1' if img2 else 'img2'} is None")
    detector = cv2.ORB_create()
    _, descriptors1 = detector.detectAndCompute(img1, None)
    _, descriptors2 = detector.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    if descriptors1 is None or descriptors2 is None:
        return 0
    matches = matcher.match(descriptors1, descriptors2)
    distance = sum([match.distance for match in matches]) / len(matches)
    return distance


def composite(output_dir, combs, select_num, scale_range, overlap_range, final_size, do_scale, do_angle):
    """
    composite images


    :param output_dir: output directory
    :param combs: all combinations (2D array, each row is a group of images, each value in group is a file path of an image)
    :param select_num: number of selected images
    :param scale_range: scale range, e.g., (0.5, 1.5)
    :param overlap_range: overlap range, e.g., (0, 0.3)
    :param final_size: final image size
    :param do_scale: whether to scale
    :param do_angle: whether to rotate
    :return: None
    """

    labels = sorted([label for _, label in combs[0]])
    # print("\r", labels, end="")
    composite_imgs = {}
    components = {}
    components_labels = {}
    for comb in combs:
        centers = []
        # calculate the center of each object
        polys = geometry.Polygon()
        np.random.shuffle(comb)
        for img_file, _ in comb:
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            if do_scale:
                img = scale(img, scale_range)
            if do_angle:
                img, _ = angle(img)
            contours, _ = cv2.findContours(img[:, :, 3], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            points = []
            for point in contours[0]:
                points.append([point[0][1], point[0][0]])
            center, poly = get_center_random(polys, geometry.Polygon(points), img.shape[0:2], overlap_range)
            polys = polys.union(poly)
            centers.append([center, img])

        # paste images on the background
        if len(centers) == 0:
            final_img = get_background(final_size, final_size)
        else:
            translation_x = min([center[0] - math.floor(img.shape[0] / 2) for center, img in centers])
            translation_y = min([center[1] - math.floor(img.shape[1] / 2) for center, img in centers])
            max_x = max([center[0] + math.ceil(img.shape[0] / 2) - translation_x for center, img in centers])
            max_y = max([center[1] + math.ceil(img.shape[1] / 2) - translation_y for center, img in centers])

            final_img = get_background(max_x, max_y)
            for center, img in centers:
                paste_img(img, final_img, center, translation_x, translation_y)

            if final_img.shape[0] > final_img.shape[1]:
                final_size_x = final_size
                final_size_y = int(final_size / final_img.shape[0] * final_img.shape[1])
            else:
                final_size_x = int(final_size / final_img.shape[1] * final_img.shape[0])
                final_size_y = final_size
            final_img = cv2.resize(final_img, (final_size_y, final_size_x))

            while True:
                id = '%05d' % random.randint(10000, 99999)
                if id not in composite_imgs:
                    break
            composite_imgs[id] = final_img

            if DO_COMPONENTS:
                components[id] = {}
                components_labels[id] = {}
                for i in range(1, len(centers)):
                    for comb in combinations(centers, i):
                        tmp_img = get_background(max_x, max_y)
                        for center, img in comb:
                            paste_img(img, tmp_img, center, translation_x, translation_y)
                        idxes = [centers.index([center, img]) for center, img in comb]
                        components[id]["".join(sorted([str(idx) for idx in idxes]))] = tmp_img
                        components_labels[id]["".join(sorted([str(idx) for idx in idxes]))] = [labels[idx] for idx in idxes]

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sample_and_save(output_dir, composite_imgs, labels, select_num, components, components_labels)
    return


def sample_and_save(output_dir, composite_imgs, labels, select_num, components=None, components_labels=None):
    """
    sample images

    :param output_dir: output directory
    :param composite_imgs: composite images
    :param labels: labels
    :param select_num: number of selected images
    :param components: components
    :param components_labels: components labels
    :return: None
    """
    # sample
    if len(composite_imgs) > select_num:
        distances = {id: defaultdict(dict) for id in composite_imgs.keys()}
        max_distance = 0
        max_distance_ids = []
        for id1, id2 in combinations(composite_imgs.keys(), 2):
            # id1, id2 = id1, id2 if id1 < id2 else id2, id1
            distance = compare(composite_imgs[id1], composite_imgs[id2])
            distances[id1][id2] = distance
            distances[id2][id1] = distance
            if distance > max_distance:
                max_distance = distance
                max_distance_ids = [id1, id2]

        selected_ids = max_distance_ids
        for _ in range(select_num - 2):
            max_distance = 0
            max_distance_id = ""
            for id in composite_imgs.keys():
                if id in selected_ids:
                    continue
                distance = sum([distances[id][id_] for id_ in selected_ids]) / len(selected_ids)
                if distance > max_distance:
                    max_distance = distance
                    max_distance_id = id
            selected_ids.append(max_distance_id)
    else:
        selected_ids = composite_imgs.keys()

    # save images
    for id in selected_ids:
        image = composite_imgs[id]
        # id = '%05d' % random.randint(10000, 99999)
        filename = f"{int(time.time() * 1000)}_{id}.png"
        cv2.imwrite(os.path.join(output_dir, filename), image)
        with open(os.path.join(output_dir, "info.csv"), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "labels"])
            writer.writerow({"filename": filename, "labels": "|".join(labels)})

        if DO_COMPONENTS:
            components_name = {}
            for idx, img in components[id].items():
                cv2.imwrite(os.path.join(output_dir, "components", f"{filename.split('.')[0]}-{idx}.png"), img)
                components_name[idx] = f"{filename.split('.')[0]}-{idx}.png"
            with open(os.path.join(output_dir, "components", "info.csv"), "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "labels"])
                rows = [
                    {"filename": components_name[idx], "labels": "|".join(sorted(components_labels[id][idx]))}
                    for idx in components[id].keys()
                ]
                writer.writerows(rows)
    return


def img_composite(
        covering_array_file, input_dir, output_dir, num, sample_times,
        max_times, scale_range, overlap_range, final_size,
        do_scale=False, do_angle=False, select_order="random",
        model=None,
):
    """
    main def for combining images

    :param covering_array_file: file path of covering array
    :param input_dir: directory of object images
    :param output_dir: directory of composite images
    :param num: number of images in each test case
    :param sample_times: sample times for each test case
    :param max_times: maximum select times for each object image
    :param scale_range: scale range
    :param overlap_range: overlap range
    :param final_size: final size of the image
    :param do_scale: whether to scale
    :param do_angle: whether to rotate
    :param select_order: select order, random or order
    :param model: model name if not None, select images for specified model
    :return: None
    """
    if os.path.exists(output_dir):
        print(f"{output_dir} exists")
        return
    matting_img, label = get_matting_img(input_dir, model)
    # print(matting_img)
    names = output_dir.split(os.sep)
    array = pd.read_csv(covering_array_file, usecols=range(label))
    tqdm.pandas(
        desc=f"{threading.currentThread().name}({names[-2]}-{names[-1]}): Selecting matting image for every line...",
        mininterval=10
    )
    start = time.process_time()
    array["img"] = array.progress_apply(
        lambda x: select(matting_img, x, num * sample_times, max_times, select_order),
        axis=1,
    )  # select sample_times*num combinations and than sample num composite images
    end = time.process_time()
    print(f"selected, time:{end - start}")
    start = time.process_time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "info.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "labels"])
        writer.writeheader()
    if DO_COMPONENTS:
        if not os.path.exists(os.path.join(output_dir, "components")):
            os.makedirs(os.path.join(output_dir, "components"))
        with open(os.path.join(output_dir, "components", "info.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "labels"])
            writer.writeheader()
    # composite, sample & save
    tqdm.pandas(
        desc=f"{threading.currentThread().name}({names[-2]}-{names[-1]}): Combining...",
        mininterval=10
    )
    array.progress_apply(
        lambda x: composite(
            output_dir, x["img"], num, scale_range, overlap_range, final_size, do_scale, do_angle
        ),
        axis=1,
    )
    end = time.process_time()
    print(f"\ncomposite done, time:{end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="composite images")
    parser.add_argument(
        "--demo", "-d",
        type=str2bool,
        default=False,
    )
    args = parser.parse_args()
    if args.demo:
        ca_file = glob.glob(os.path.join(COVERING_ARRAY_DIR, "adaptive random", "ca_adaptive random*.csv"))[0]
        input_dir = os.path.join(MATTING_IMG_DIRS["VOC"])
        model = "msrn"
        output_dir = os.path.join(f"{COMPOSITE_IMG_DIRS['VOC']}_{VERSION}", model, "adaptive random_6_3_2_No1")
        img_composite(
            covering_array_file=ca_file,
            input_dir=input_dir,
            output_dir=output_dir,
            num=10, sample_times=1, max_times=1,
            scale_range=(0.5, 1.5), overlap_range=(0, 0.3), final_size=640,
            do_scale=DO_SCALE, do_angle=DO_ANGLE,
            select_order="random",
            model=model,
        )
    else:
        thread_pool = ThreadPoolExecutor(max_workers=16)
        for dataname in DATA_NAMES:
            matting_img_dir = MATTING_IMG_DIRS[dataname]
            composite_img_dir = COMPOSITE_IMG_DIRS[dataname]
            classes = composite_img_dir.split('_')[-1]
            tasks = [
                # ca_method_classes_k_tau
                f"{CA_METHOD}_{classes}_{k}_{tau}" for k in [4] for tau in [2]
            ]
            for task in tasks:
                files = glob.glob(os.path.join(COVERING_ARRAY_DIR, task.split("_")[0], f"ca_{task}*.csv"))
                for i, file in enumerate(files):
                    input_dir = matting_img_dir
                    for model in MODELS[dataname]:
                        # start = time.process_time()
                        print(f"task \"{model} {task}\" of {matting_img_dir}, {file} No{i+1} start")
                        output_dir = os.path.join(f"{composite_img_dir}_{VERSION}", model, f"{task}_No{i+1}")
                        thread_pool.submit(
                            img_composite,
                            covering_array_file=file,
                            input_dir=input_dir,
                            output_dir=output_dir,
                            num=10, sample_times=3, max_times=1,
                            scale_range=(0.5, 1.5), overlap_range=(0, 0.3), final_size=640,
                            do_scale=DO_SCALE, do_angle=DO_ANGLE,
                            select_order="random",
                            model=model,
                        )
        thread_pool.shutdown(wait=True)
