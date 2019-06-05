import torch
import os
import tqdm
import argparse
import time
import numpy as np
from itertools import chain
from glob import glob
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from neural_point_splatter.neuralSplatter import NeuralPointSplatter, BaselineRenderer, CameraSampler
from neural_point_splatter.mathHelper import dot
from neural_point_splatter.splatterIo import saveAsPng, readScene, readCloud, getBasename, checkScenePaths, writeCameras
from demos.app_skeletton import renderScene, parse_device, writeScene
from pytorch_points.network.operations import normalize_point_batch, batch_normals
from pytorch_points.utils.pc_utils import save_ply

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Render a given point cloud.')
    parser.add_argument("--input", nargs=1, help="paths to input")
    parser.add_argument("--target", nargs=1, help="paths to target")
    parser.add_argument('-s', '--scene', nargs=1, default=["../example_data/scenes/template.json"],
                        help='Input file')
    parser.add_argument('-o', '--output', dest='output',
                        help='Output file path')
    parser.add_argument('--width', dest='image_width', type=int, default=None,
                        help='Desired image width in pixels.')
    parser.add_argument('--height', dest='image_height', type=int, default=None,
                        help='Desired image height in pixels.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                        help='If true: show additional output like inbetween calculations')
    parser.add_argument('-d', '--device', dest='device', default='cuda:0', help='Device to run the computations on, options: cpu, cuda')
    parser.add_argument('-c', '--gen-camera', type=int, default=0, help='number of random cameras')
    parser.add_argument('-cO', '--cam-offset', dest="gen_camera_offset", type=float, nargs=2, default=[5, 20], help='depth offset for generated cameras')
    parser.add_argument('-cF', '--cam-focal', dest="gen_camera_focal", type=float, default=15, help='focal length for generated cameras')
    parser.add_argument('--cutoff', type=float, default=1, help='cutoff threshold')
    parser.add_argument('--baseline', action="store_true", help="use baseline depth renderer")
    parser.add_argument('-k', '--topK', dest='topK', type=int, default=5, help='topK for merging depth')
    parser.add_argument('-mT', '--merge_threshold', type=float, default=0.05, help='threshold for merging depth')
    parser.add_argument('--vrk-mode', help="nearestNeighbor or constant", choices=["nearestNeighbor", "constant"], default="constant")
    parser.add_argument('--pca-normal', action="store_true", help="recompute noisy point cloud normal with pca")
    parser.add_argument('--name', type=str, default="*.ply")

    args = parser.parse_args()
    args.input = args.input.pop()
    args.target = args.target.pop()
    target_points_paths = glob(os.path.join(args.target, "**", args.name), recursive=True)
    input_points_paths = glob(os.path.join(args.input, "**", args.name), recursive=True)
    assert(len(target_points_paths) == len(input_points_paths))

    if args.output is None:
        args.output = 'renders/'
    VERBOSE = args.verbose

    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)

    (device, isCpu) = parse_device(args)
    torch.cuda.set_device(device)

    if VERBOSE:
        print("Rendering on:", device)

    if args.baseline:
        MODEL = BaselineRenderer
    else:
        MODEL = NeuralPointSplatter

    scenePath = checkScenePaths(args.scene).pop()
    scene = readScene(scenePath, device=device)
    scene.cloud.VrkMode = args.vrk_mode
    with torch.no_grad():
        splatter = MODEL(scene, device=device, verbose=False, shading=scene.cloud.shading,
                         mergeTopK=args.topK, mergeThreshold=args.merge_threshold, cutOffThreshold=args.cutoff)
        for inputPath, targetPath in zip(input_points_paths, target_points_paths):
            inputRelPath = os.path.relpath(inputPath, args.input)
            targetRelPath = os.path.relpath(targetPath, args.target)
            input_outDir = os.path.join(args.output, "input_rendered", inputRelPath[:-4])
            target_outDir = os.path.join(args.output, "target_rendered", targetRelPath[:-4])

            readSceneTick = time.time()
            targetPoints = readCloud(targetPath, device=device)
            # targetPoints_, _, _ = normalize_point_batch(targetPoints[:, :, :3], NCHW=False)
            # targetPoints_.squeeze_(0)
            # targetPoints[:, :3] = targetPoints_
            inputPoints = readCloud(inputPath, device=device)
            # inputPoints_, _, _ = normalize_point_batch(inputPoints[:, :, :3], NCHW=False)
            # inputPoints.squeeze_(0)
            # inputPoints[:, :3] = inputPoints_
            if args.pca_normal:
                inputNormals = batch_normals(inputPoints[:, :3].unsqueeze(0), nn_size=32, NCHW=False)
                inputNormals = torch.where(dot(inputNormals, targetPoints[:, 3:6].unsqueeze(0), dim=-1).unsqueeze(-1) < 0, -inputNormals, inputNormals)
                inputPoints[:, 3:6] = inputNormals.squeeze(0)
                # targetNormals = batch_normals(targetPoints[:, :3].unsqueeze(0), nn_size=32, NCHW=False)
                # targetNormals = torch.where(dot(targetNormals, targetPoints[:, 3:6].unsqueeze(0), dim=-1).unsqueeze(-1) < 0, -targetNormals, targetNormals)
                # targetPoints[:, 3:6] = targetNormals.squeeze(0)
                # save_ply(targetPoints[:, :3].cpu().numpy(), targetPath[:-4]+"_pca.ply", normals=targetPoints[:, 3:6].cpu().numpy())
                save_ply(inputPoints[:, :3].cpu().numpy(), inputPath[:-4]+"_pca.ply", normals=inputPoints[:, 3:6].cpu().numpy())

            readSceneTock = time.time()
            renderCount = 0

            # for offset in range(int(args.gen_camera_offset[0]), int(args.gen_camera_offset[1])):
            #     offsets = (np.clip(np.random.randn(2), -0.2, 0.2)+1)*offset
            #     for o in offsets:
            #         if args.gen_camera > 0:
            #             camSampler = CameraSampler(args.gen_camera, o, args.gen_camera_focal,
            #                                        points=targetPoints[:, :3].unsqueeze(0),
            #                                        camWidth=args.image_width,
            #                                        camHeight=args.image_height,
            #                                        filename="../example_data/pointclouds/dome_300.ply")
            #             cameras = []
            #             for i in range(args.gen_camera):
            #                 cam = next(camSampler)
            #                 cameras.append(cam)
            #         else:
            #             cameras = None

            #         if cameras is not None:
            #             splatter.initCameras(cameras=cameras)
            #             # writeCameras(scene, args.output + '/cameras.ply')
            #         else:
            #             splatter.initCameras(cameras=scene.cameras)

            #         rendered = []
            #         for i, cam in enumerate(scene.cameras):
            #             try:
            #                 splatter.setCamera(i)
            #                 scene.loadPoints(inputPoints)
            #                 splatter.setCloud(scene.cloud)
            #                 inputRendered = splatter.render().detach()[0]
            #                 scene.loadPoints(targetPoints)
            #                 splatter.setCloud(scene.cloud)
            #                 targetRendered = splatter.render().detach()[0]
            #             except Exception as e:
            #                 print(inputPath, targetPath, renderCount, e)
            #             else:
            #                 if VERBOSE and i == 0:
            #                     pngDir = os.path.join(input_outDir, "png")
            #                     os.makedirs(os.path.join(input_outDir, "png"), exist_ok=True)
            #                     saveAsPng(inputRendered.cpu(), os.path.join(pngDir, 'cam%04d_input.png' % renderCount))
            #                 np.save(os.path.join(input_outDir, 'cam%04d_input.npy' % renderCount), inputRendered.cpu().numpy())
            #                 if VERBOSE and i == 0:
            #                     pngDir = os.path.join(target_outDir, "png")
            #                     os.makedirs(os.path.join(target_outDir, "png"), exist_ok=True)
            #                     saveAsPng(targetRendered.cpu(), os.path.join(pngDir, 'cam%04d_target.png' % (renderCount)))
            #                 np.save(os.path.join(target_outDir, 'cam%04d_target.npy' % renderCount), targetRendered.cpu().numpy())
            #                 renderCount += 1

            print(inputPath, targetPath)
