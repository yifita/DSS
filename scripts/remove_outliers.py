import torch
import math
import png
import sys
import os
import numpy as np
import time
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from neural_point_splatter.splatterIo import readCloud, readScene, saveAsPng, readImage, writeScene, writeCameras
from neural_point_splatter.neuralSplatter import NeuralPointSplatter, BaselineRenderer, CameraSampler, viewFromError, removeOutlier
from app_skeletton import argument_device, parse_device, argument_verbose
from pytorch_points.utils.pc_utils import save_ply_property


def runRemoveOutlier(scene, gtpointPath, predpointPath, keyName, genCamera,
                     genCameraOffset, genCameraFocalLength, noiseSigma=50,
                     mergeThreshold=0.05, mergeTopK=5, cutOff=1.0, device=None, verbose=False,
                     ):
    splatter = NeuralPointSplatter(scene, device=device,
                                   shading=scene.cloud.shading,
                                   cutOffThreshold=cutOff,
                                   mergeThreshold=mergeThreshold,
                                   mergeTopK=mergeTopK)
    predPoints = readCloud(predpointPath, device="cpu")
    gtPoints = readCloud(gtpointPath, device="cpu")

    if genCamera > 0:
        camSampler = CameraSampler(genCamera, genCameraOffset, genCameraFocalLength, points=scene.cloud.localPoints,
                                   camWidth=128, camHeight=128, filename="../example_data/pointclouds/sphere_300.ply")
        cameras = []
        for i, cam in enumerate(camSampler):
            cameras.append(cam)

        splatter.initCameras(cameras)

    scene.loadPoints(gtPoints)
    splatter.setCloud(scene.cloud)
    gtImages = []
    for i in range(len(cameras)):
        splatter.setCamera(i)
        gt = splatter.render()[0]
        gtImages.append(gt)

    scores = [None]*len(cameras)
    scene.loadPoints(predPoints)
    splatter.setCloud(scene.cloud)
    for i, gt in enumerate(gtImages):
        if gt is None:
            continue
        splatter.setCamera(i)
        splatter.convertToCameraSpace()
        projectedPoints = splatter.camera.projectPoints(splatter.cameraPoints)
        score = removeOutlier(gt, projectedPoints, sigma=noiseSigma)
        if verbose:
            saveAsPng(gt.cpu(), keyName + '_clean_gt_cam%d.png' % i)
            save_ply_property(splatter.localPoints.cpu()[0], 1-score.cpu(), keyName+'_clean_score_cam%d.ply' % i, property_max=1,
                              normals=splatter.localNormals.cpu()[0], cmap_name="gnuplot2")
            rendered = splatter.render()[0]
            saveAsPng(rendered.cpu(), keyName + '_clean_pred_cam%d.png' % i)
        scores[i] = score

    scores = torch.stack(scores, dim=0)
    scores = torch.prod(scores, dim=0)
    # import pdb
    # pdb.set_trace()
    # some reduction
    # indices = torch.nonzero(score > 1e-4).squeeze()
    _, indices = torch.topk(scores, int(splatter.localPoints.shape[1]*0.99), dim=0)
    newPoints = torch.index_select(splatter.localPoints.data, 1, indices)
    newNormals = torch.index_select(splatter.localNormals.data, 1, indices)
    newColors = torch.index_select(splatter.pointColors.data, 1, indices)
    scene.cloud.localPoints = newPoints[0]
    scene.cloud.localNormals = newNormals[0]
    scene.cloud.color = newColors[0]
    writeScene(scene, keyName + '_cleaned_scene.json', keyName + '_cleaned_cloud.ply')


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Demonstrate that we can learn a color.')
    parser.add_argument('source', metavar="source", nargs='?',
                        default="../example_data/scenes/grid16.json",
                        help='Souce scene/ground truth used for learning: it will be modified by a modifier.')
    parser.add_argument('-t', "--target", dest="gtPath", nargs=1,
                        help='target point cloud')
    parser.add_argument('-i', "--input", dest="predPath", nargs=1,
                        help='input point cloud')
    parser.add_argument('-o', '--output', dest='output', help='Output file name (without extension)')
    parser.add_argument('-c', '--gen-camera', type=int, default=0, help='number of random cameras')
    parser.add_argument('-cO', '--camOffset', dest="gen_camera_offset", type=float, default=20, help='depth offset for generated cameras')
    parser.add_argument('-cF', '--camFocal', dest="gen_camera_focal", type=float, default=15, help='focal length for generated cameras')
    parser.add_argument('--cutoff', type=float, default=1, help='cutoff threshold')
    parser.add_argument('-k', '--topK', dest='topK', type=int, default=5, help='topK for merging depth')
    parser.add_argument('-mT', '--merge_threshold', type=float, default=0.05, help='threshold for merging depth')
    parser.add_argument('--noise_sigma', type=float, default=50, help='noise sigma')
    argument_device(parser)
    args = parser.parse_args()

    predPath = args.predPath.pop()
    gtPath = args.gtPath.pop()
    if args.output is None:
        args.output = 'learn_example/learn_example_'

    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)
    if args.output is None:
        args.output = 'learn_example/learn_example_'
    (device, isCpu) = parse_device(args)
    torch.cuda.set_device(device)
    keyName = args.output + "/shape"

    os.makedirs(os.path.dirname(keyName), exist_ok=True)
    # Create ground truth
    scene = readScene(args.source, device="cpu")
    runRemoveOutlier(scene, gtPath, predPath, keyName, genCamera=args.gen_camera,
                     genCameraOffset=args.gen_camera_offset, genCameraFocalLength=args.gen_camera_focal,
                     noiseSigma=args.noise_sigma,
                     mergeThreshold=args.merge_threshold, mergeTopK=args.topK, cutOff=args.cutoff,
                     device=device, verbose=args.verbose)
