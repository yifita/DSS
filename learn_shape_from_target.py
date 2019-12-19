import torch
import math
import sys
import os
import numpy as np
import time
from DSS.utils.splatterIo import readScene, saveAsPng, writeScene, writeCameras
from DSS.utils.trainer import Trainer
from DSS.options.deformation_options import DeformationOptions
from DSS.core.renderer import saved_variables as tmp_saved_v


def trainShapeOnImage(scene, refScene, opt, baseline=False):
    expr_dir = os.path.join(opt.output, opt.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)

    trainer = Trainer(opt, scene)
    trainer.setup(opt, scene.cloud)

    logInterval = math.floor(1+sum(opt.steps)//20)
    renderForwardTime = 0.0
    lossTime = 0.0
    optimizerStep = 0.0

    log_variables = {}
    writeScene(refScene, os.path.join(expr_dir, 't000_scene_gt.json'), os.path.join(expr_dir, "gt.ply"))
    with open(expr_dir + "/loss.csv", 'w') as loss_log:
        for c in range(opt.cycles):
            # creat new reference
            tb = c*sum(opt.steps)+opt.startingStep
            te = (c+1)*sum(opt.steps)+opt.startingStep
            t = tb
            with torch.no_grad():
                trainer.create_reference(refScene)
                writeScene(refScene, os.path.join(
                    expr_dir, 't%03d_scene_gt.json' % t))
                writeCameras(refScene, os.path.join(
                    expr_dir, 't%03d_cameras.ply' % t))
                for i, gt in enumerate(trainer.groundtruths):
                    saveAsPng(gt.cpu()[0], os.path.join(
                        expr_dir, 't%03d_cam%d_gt.png' % (t, i)))
                trainer.initiate_cycle()

            for t in range(tb, te):
                if t % logInterval == 0:
                    writeScene(scene, os.path.join(
                        expr_dir, 't%03d_scene.json' % t), os.path.join(expr_dir, "t%03d.ply" % t))

                trainer.optimize_parameters()
                for k in tmp_saved_v:
                    if k == "renderable_idx":
                        continue
                    if k not in log_variables:
                        log_variables[k] = tmp_saved_v[k].detach().numpy()
                    else:
                        log_variables[k] = np.concatenate(
                            [log_variables[k], tmp_saved_v[k].detach().numpy()], axis=0)



                if t % logInterval == 0:
                    for i, prediction in enumerate(trainer.predictions):
                        saveAsPng(prediction.detach().cpu()[0], os.path.join(
                            expr_dir, 't%03d_cam%d' % (t, i) + ".png"))

                loss_str = ",".join(
                    ["%.3f" % v for v in trainer.loss_image])
                reg_str = ",".join(["%.3f" % v for v in trainer.loss_reg])
                entries = [trainer.modifier] + [loss_str] + [reg_str]
                loss_log.write(",".join(entries)+"\n")
                print("{:03d} {}: lr {} loss ({}) \n         :       reg ({})".format(
                    t, trainer.modifier, trainer.lr, loss_str, reg_str))

            trainer.finish_cycle()

    writeScene(scene, os.path.join(expr_dir, 'final_scene.json'),
               os.path.join(expr_dir, 'final_cloud.ply'))
    np.save(os.path.join(expr_dir, "log_variables"), log_variables)


if __name__ == "__main__":
    opt = DeformationOptions().parse()

    # reproducability
    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)

    # load scenes
    refScene = readScene(opt.ref, device="cpu")
    scene = readScene(opt.source, device="cpu")

    scene.cloud.shading = refScene.cloud.shading
    scene.pointlightPositions = refScene.pointlightPositions
    scene.pointlightColors = refScene.pointlightColors
    scene.sunDirections = refScene.sunDirections
    scene.sunColors = refScene.sunColors
    scene.ambientLight = refScene.ambientLight

    trainShapeOnImage(scene, refScene, opt,
                      baseline=opt.baseline)
