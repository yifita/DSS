import torch
import os
import numpy as np
import time
from DSS.utils.splatterIo import readScene, saveAsPng, writeScene, writeCameras
from DSS.utils.trainer import Trainer, viewFromError, removeOutlier
from DSS.options.finetune_options import FinetuneOptions
from pytorch_points.utils.pc_utils import save_ply_property


def normal_length_normalizer(normal, criterion):
    squaredNorm = torch.sum(normal**2, dim=-1)
    return criterion(squaredNorm, torch.ones_like(squaredNorm))


def trainShapeOnImage(scene, refScene, opt, baseline=False, benchmark=False):
    expr_dir = os.path.join(opt.output, opt.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)

    trainer = Trainer(opt, scene)
    trainer.setup(opt, scene.cloud)

    logInterval = (1+sum(opt.steps))//20
    renderForwardTime = 0.0
    lossTime = 0.0
    optimizerStep = 0.0

    with open(os.path.join(expr_dir, "loss.csv"), 'w') as loss_log:
        learnTick = time.time()
        for c in range(opt.cycles):
            # creat new reference
            tb = c*sum(opt.steps)+opt.startingStep
            te = (c+1)*sum(opt.steps)+opt.startingStep
            t = tb

            with torch.no_grad():
                # render reference
                trainer.create_reference(refScene)

                # render prediction
                maxDiffs = []
                selectedCameras = []
                for i in range(len(refScene.cameras)):
                    trainer.forward(i)
                    prediction = trainer.predictions[i]
                    maxDiff, selectedCamera = viewFromError(opt.genCamera, trainer.groundtruths[i][0], prediction.detach()[0],
                                                            trainer.model._localPoints.detach(), trainer.model._projPoints.detach(),
                                                            trainer.model,
                                                            offset=opt.camOffset*(0.997**c))
                    maxDiffs.append(maxDiff)
                    selectedCameras.append(selectedCamera)

                maxId = torch.stack(maxDiffs, dim=0).argmax()
                selectedCamera = selectedCameras[maxId]
                # render again
                trainer.create_reference(refScene, selectedCamera)
                writeScene(refScene, os.path.join(expr_dir, 't%03d_scene_gt.json' % t))
                writeCameras(refScene, os.path.join(expr_dir, 't%03d_cameras.ply' % t))
                for i, gt in enumerate(trainer.groundtruths):
                    if gt is not None:
                        saveAsPng(gt.cpu()[0], os.path.join(expr_dir, 't%03d_cam%d_gt.png' % (t, i)))
                trainer.initiate_cycle()

            for t in range(tb, te):
                if t % logInterval == 0 and not benchmark:
                    writeScene(scene, os.path.join(expr_dir, 't%03d_scene.json' % t), os.path.join(expr_dir, "t%03d.ply" % t))

                trainer.optimize_parameters()

                if t % logInterval == 0 and not benchmark:
                    for i, prediction in enumerate(trainer.predictions):
                        saveAsPng(prediction.detach().cpu()[0], os.path.join(expr_dir, 't%03d_cam%d' % (t, i) + ".png"))

                if not benchmark:
                    loss_str = ",".join(["%.3f" % v for v in trainer.loss_image])
                    reg_str = ",".join(["%.3f" % v for v in trainer.loss_reg])
                    entries = [trainer.modifier] + [loss_str] + [reg_str]
                    loss_log.write(",".join(entries)+"\n")
                    print("{:03d} {}: lr {} loss ({}) \n         :       reg ({})".format(
                        t, trainer.modifier, trainer.lr, loss_str, reg_str))

            trainer.finish_cycle()

    # outlier removal
    with torch.no_grad():
        # re-project
        scores = []
        for i, gt in enumerate(trainer.groundtruths):
            if gt is None:
                continue
            trainer.model.setCamera(i)
            trainer.model.convertToCameraSpace()
            projectedPoints = trainer.model.camera.projectPoints(trainer.model.cameraPoints)
            score = removeOutlier(gt, projectedPoints, sigma=100)

            saveAsPng(gt.cpu()[0], os.path.join(expr_dir, 'clean_gt_cam%d.png' % i))
            save_ply_property(trainer.model.localPoints.cpu()[0].detach(), 1-score.cpu(), os.path.join(expr_dir, 'clean_score_cam % d.ply' % i), property_max=1,
                              normals=trainer.model.localNormals.detach().cpu()[0], cmap_name="gnuplot2")
            rendered = trainer.model.render()[0]
            saveAsPng(rendered.cpu(), os.path.join(expr_dir, 'clean_pred_cam%d.png' % i))
            scores.append(score)

        scores = torch.stack(scores, dim=0)
        scores = torch.prod(scores, dim=0)
        _, indices = torch.topk(scores, int(trainer.model.localPoints.shape[1]*0.99), dim=0)
        # _, indices = torch.topk(score, int(trainer.model.localPoints.shape[1]*0.99), dim=0)
        newPoints = torch.index_select(trainer.model.localPoints.data, 1, indices)
        newNormals = torch.index_select(trainer.model.localNormals.data, 1, indices)
        newColors = torch.index_select(trainer.model.pointColors.data, 1, indices)

        scene.cloud.localPoints = newPoints[0]
        scene.cloud.localNormals = newNormals[0]
        scene.cloud.color = newColors[0]

    writeScene(scene, os.path.join(expr_dir, 'final_scene.json'), os.path.join(expr_dir, 'final_cloud.ply'))


if __name__ == "__main__":
    opt = FinetuneOptions().parse()

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

    trainShapeOnImage(scene, refScene, opt, baseline=opt.baseline, benchmark=opt.benchmark)
