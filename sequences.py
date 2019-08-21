"""render a point cloud in 360 degree"""
from __future__ import division, print_function
import torch
import os
import argparse
import time
import numpy as np
from itertools import chain
from glob import glob
import sys
from DSS.utils.matrixConstruction import rotationMatrixY, rotationMatrixX, rotationMatrixZ, batchAffineMatrix
from DSS.utils.splatterIo import saveAsPng, readScene, readCloud, getBasename, writeCameras, writeScene
from DSS.core.renderer import createSplatter
from DSS.core.camera import CameraSampler
from DSS.options.render_options import RenderOptions


def rotMatrix(axis):
    if axis.lower() == "x":
        return rotationMatrixY
    elif axis.lower() == "y":
        return rotationMatrixY
    else:
        return rotationMatrixZ


if __name__ == "__main__":
    opt = RenderOptions().parse()
    points_paths = list(chain.from_iterable(glob(p) for p in opt.points))
    assert(len(points_paths) > 0), "Found no point clouds with path {}".format(points_paths)
    points_relpaths = None
    if len(points_paths) > 1:
        points_dir = os.path.commonpath(points_paths)
        points_relpaths = [os.path.relpath(p, points_dir) for p in points_paths]
    else:
        points_relpaths = [os.path.basename(p) for p in points_paths]

    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)

    scene = readScene(opt.source, device="cpu")

    getRotationMatrix = rotMatrix(opt.rot_axis)
    with torch.no_grad():
        splatter = createSplatter(opt, scene=scene)

        for i in range(len(scene.cameras)):
            scene.cameras[i].width = opt.width
            scene.cameras[i].height = opt.height
            # scene.cameras[i].focalLength = opt.camFocalLength

        splatter.initCameras(cameras=scene.cameras)

        for pointPath, pointRelPath in zip(points_paths, points_relpaths):
            keyName = os.path.join(opt.output, pointRelPath[:-4])
            print(pointPath)
            points = readCloud(pointPath, device="cpu")
            fileName = getBasename(pointPath)
            # find point center
            center = torch.mean(points[:, :3], dim=0, keepdim=True)
            points[:, :3] -= center
            scene.loadPoints(points)
            splatter.setCloud(scene.cloud)
            splatter.pointPosition.data.copy_(center)
            for i, cam in enumerate(scene.cameras):
                # compute object rotation
                cnt = 0
                for ang in range(0, 360, 3):
                    rot = getRotationMatrix(torch.tensor(ang*np.pi/180).to(device=splatter.pointRotation.device))
                    splatter.pointRotation.data.copy_(rot.unsqueeze(0))
                    splatter.m2w = batchAffineMatrix(splatter.pointRotation, splatter.pointPosition, splatter.pointScale)

                    # set camera to look at the center
                    splatter.setCamera(i)
                    result = splatter.render()
                    if result is None:
                        continue
                    result = result.detach()[0]

                    if splatter.shading == "albedo":
                        cmax = 1
                        saveAsPng(result.cpu(), keyName + '_cam%02d_%03d.png' % (i, cnt), cmin=0, cmax=cmax)
                    else:
                        saveAsPng(result.cpu(), keyName + '_cam%02d_%03d.png' % (i, cnt), cmin=0)

                    cnt += 1

            print(pointRelPath)
