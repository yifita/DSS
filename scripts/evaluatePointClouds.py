import argparse
import os
import numpy as np
import torch
from glob import glob
import re
import csv
from collections import OrderedDict
from pytorch_points.network.operations import normalize_point_batch, group_knn
from pytorch_points.network.model_loss import nndistance
from pytorch_points.utils.pc_utils import save_ply_property, load, save_ply_property


def get_filenames(source, extension):
    # If extension is a list
    if source is None:
        return []
    # Seamlessy load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source):
            if not isinstance(extension, str):
                for fmt in extension:
                    source_fns += get_filenames(source, fmt)
            else:
                source_fns = sorted(
                    glob("{}/**/*{}".format(source, extension), recursive=True))
        elif os.path.isfile(source):
            source_fns = [source]
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(get_filenames(s, extension=extension))
    return source_fns


parser = argparse.ArgumentParser()
parser.add_argument("--gt", type=str, required=True, help="directory or file name for ground truth point clouds")
parser.add_argument("--pred", type=str, nargs="+", required=True, help="directorie of predictions")
parser.add_argument("--name", type=str, required=True, help="name pattern if provided directory for pred and gt")
FLAGS = parser.parse_args()
if os.path.isdir(FLAGS.gt):
    GT_DIR = FLAGS.gt
    gt_paths = get_filenames(GT_DIR, ("ply", "pcd", "xyz"))
    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
elif os.path.isfile(FLAGS.gt):
    gt_paths = [FLAGS.gt]

PRED_DIR = FLAGS.pred
NAME = FLAGS.name


fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"] + ["nuc_%d" % d for d in range(7)]
print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
for D in PRED_DIR:
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    counter = 0
    pred_paths = glob(os.path.join(D, "**", NAME), recursive=True)
    if len(pred_paths) == 1 and len(pred_paths) > 1:
        gt_pred_pairs = []
        for p in pred_paths:
            name, ext = os.path.splitext(os.path.basename(p))
            assert(ext in (".ply", ".xyz"))
            try:
                gt = gt_paths[gt_names.index(name)]
            except ValueError:
                pass
            else:
                gt_pred_pairs.append((gt, p))
    else:
        gt_pred_pairs = []
        for p in pred_paths:
            gt_pred_pairs.append((gt_paths[0], p))

    # print("total inputs ", len(gt_pred_pairs))
    # tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
    tag = os.path.basename(os.path.dirname(gt_pred_pairs[0][1]))

    print("{:60s}".format(tag), end=' ')
    global_p2f = []
    global_density = []
    with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        for gt_path, pred_path in gt_pred_pairs:
            row = {}
            gt = load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            pred = load(pred_path)
            pred = pred[:, :3]

            row["name"] = os.path.basename(pred_path)
            pred = pred[np.newaxis, ...]

            pred = torch.from_numpy(pred).cuda()
            gt = torch.from_numpy(gt).cuda()

            pred_tensor, centroid, furthest_distance = normalize_point_batch(pred)
            gt_tensor, centroid, furthest_distance = normalize_point_batch(gt)

            # B, P_predict, 1
            cd_forward, cd_backward = nndistance(pred, gt)
            # cd_forward, _ = knn_point(1, gt_tensor, pred_tensor)
            # cd_backward, _ = knn_point(1, pred_tensor, gt_tensor)
            # cd_forward = cd_forward[0, :, 0]
            # cd_backward = cd_backward[0, :, 0]
            cd_forward = cd_forward.detach().cpu().numpy()[0]
            cd_backward = cd_backward.detach().cpu().numpy()[0]

            save_ply_property(pred.squeeze(0).detach().cpu().numpy(), cd_forward, pred_path[:-4]+"_cdF.ply", property_max=0.003, cmap_name="jet")
            save_ply_property(gt.squeeze(0).detach().cpu().numpy(), cd_backward, pred_path[:-4]+"_cdB.ply", property_max=0.003, cmap_name="jet")

            md_value = np.mean(cd_forward)+np.mean(cd_backward)
            hd_value = np.max(np.amax(cd_forward, axis=0)+np.amax(cd_backward, axis=0))
            cd_backward = np.mean(cd_backward)
            cd_forward = np.mean(cd_forward)
            # row["CD_forward"] = np.mean(cd_forward)
            # row["CD_backwar"] = np.mean(cd_backward)
            row["CD"] = cd_forward+cd_backward

            row["hausdorff"] = hd_value
            avg_md_forward_value += cd_forward
            avg_md_backward_value += cd_backward
            avg_hd_value += hd_value
            if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
                point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                row["p2f avg"] = np.nanmean(point2mesh_distance)
                row["p2f std"] = np.nanstd(point2mesh_distance)
                global_p2f.append(point2mesh_distance)
            if os.path.isfile(pred_path[:-4] + "_density.xyz"):
                density = load(pred_path[:-4] + "_density.xyz")
                global_density.append(density)
                std = np.std(density, axis=0)
                for i in range(7):
                    row["nuc_%d" % i] = std[i]
            writer.writerow(row)
            counter += 1

        row = OrderedDict()

        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        # row["CD_forward"] = avg_md_forward_value
        # row["CD_backward"] = avg_md_backward_value
        row["CD"] = avg_md_forward_value+avg_md_backward_value
        row["hausdorff"] = avg_hd_value
        if global_p2f:
            global_p2f = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2f)
            std_p2f = np.nanstd(global_p2f)
            row["p2f avg"] = mean_p2f
            row["p2f std"] = std_p2f
        if global_density:
            global_density = np.concatenate(global_density, axis=0)
            nuc = np.std(global_density, axis=0)
            for i in range(7):
                row["nuc_%d" % i] = std[i]

        writer.writerow(row)
        print("|".join(["{:>15.8f}".format(d) for d in row.values()]))
