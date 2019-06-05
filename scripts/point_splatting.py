import torch
import os
import argparse
import time
import numpy as np
from itertools import chain
from glob import glob
from DSS.core.renderer import createSplatter
from DSS.utils.splatterIo import saveAsPng, readScene, readCloud, getBasename, writeCameras
from DSS.utils.trainer import CameraSampler
from DSS.options.render_options import RenderOptions

if __name__ == "__main__":
    opt = RenderOptions().parse()
    points_paths = list(chain.from_iterable(glob(p) for p in opt.points))
    assert(len(points_paths) > 0), "Found no point clouds in with path {}".format(points_paths)
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

    if opt.genCamera > 0:
        camSampler = CameraSampler(opt.genCamera, opt.camOffset, opt.camFocalLength, points=scene.cloud.localPoints,
                                   camWidth=opt.width, camHeight=opt.height, filename="../example_data/pointclouds/sphere_300.ply")
    with torch.no_grad():
        splatter = createSplatter(opt, scene=scene)

        if opt.genCamera > 0:
            cameras = []
            for i in range(opt.genCamera):
                cam = next(camSampler)
                cameras.append(cam)

            splatter.initCameras(cameras=cameras)
            writeCameras(scene, os.path.join(opt.output, 'cameras.ply'))
        else:
            for i in range(len(scene.cameras)):
                scene.cameras[i].width = opt.width
                scene.cameras[i].height = opt.height

        splatter.initCameras(cameras=scene.cameras)

        for pointPath, pointRelPath in zip(points_paths, points_relpaths):
            keyName = os.path.join(os.path.join(opt.output, pointRelPath[:-4]))
            readSceneTick = time.time()
            readSceneTock = time.time()
            points = readCloud(pointPath, device="cpu")
            scene.loadPoints(points)
            fileName = getBasename(pointPath)
            splatter.setCloud(scene.cloud)

            rendered = []
            for i, cam in enumerate(scene.cameras):
                splatter.setCamera(i)
                result = splatter.render()
                if result is None:
                    continue
                result = result.detach()[0]
                rendered.append(result)
            print(pointRelPath)
            for i, gt in enumerate(rendered):
                if splatter.shading == "albedo":
                    cmax = 1
                else:
                    cmax = None
                saveAsPng(gt.cpu(), keyName + '_cam%02d.png' % i, cmin=0, cmax=cmax)
            # stacked = torch.stack(rendered, dim=0)
            # np.save(keyName+'_views.npy', stacked.cpu().numpy())
