import torch
import math
import os
import numpy as np
import time
import importlib
from DSS.utils.splatterIo import readCloud, readScene, saveAsPng, writeScene
from DSS.utils.trainer import FilterTrainer as Trainer
from DSS.options.filter_options import FilterOptions
from pytorch_points.network.operations import normalize_point_batch
from pytorch_points.utils.pc_utils import load


def trainImageFilter(scene, benchmark=False):
    expr_dir = os.path.join(opt.output, opt.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)

    trainer = Trainer(opt, scene)
    trainer.setup(opt, scene.cloud)

    logInterval = math.floor(1+sum(opt.steps)//20)
    renderForwardTime = 0.0
    lossTime = 0.0
    optimizerStep = 0.0

    with torch.autograd.detect_anomaly():
        with open(os.path.join(expr_dir, "loss.csv"), 'w') as loss_log:
            for c in range(opt.cycles):
                # creat new reference
                tb = c*sum(opt.steps)+opt.startingStep
                te = (c+1)*sum(opt.steps)+opt.startingStep
                t = tb

                with torch.no_grad():
                    trainer.create_reference(scene)
                    trainer.initiate_cycle()
                    for i, pair in enumerate(zip(trainer.groundtruths, trainer.predictions)):
                        post, pre = pair
                        diff = post - pre
                        saveAsPng(pre.cpu(), os.path.join(expr_dir, 't%03d_cam%d_init.png' % (t, i)))
                        saveAsPng(post.cpu(), os.path.join(expr_dir, 't%03d_cam%d_gt.png' % (t, i)))
                        saveAsPng(diff.cpu(), os.path.join(expr_dir, 't%03d_cam%d_diff.png' % (t, i)))

                for t in range(tb, te):
                    if t % logInterval == 0 and not benchmark:
                        writeScene(scene, os.path.join(expr_dir, 't%03d' % t +
                                                       '_values.json'), os.path.join(expr_dir, 't%03d' % t + '.ply'))

                    trainer.optimize_parameters()
                    if t % logInterval == 0 and not benchmark:
                        for i, prediction in enumerate(trainer.predictions):
                            saveAsPng(prediction.detach().cpu()[0], os.path.join(expr_dir, 't%03d_cam%d' % (t, i) + ".png"))

                    if not benchmark:
                        loss_str = ",".join(["%.3f" % (100*v) for v in trainer.loss_image])
                        reg_str = ",".join(["%.3f" % (100*v) for v in trainer.loss_reg])
                        entries = [trainer.modifier] + [loss_str] + [reg_str]
                        loss_log.write(",".join(entries)+"\n")
                        print("{:03d} {}: lr {} loss ({}) \n         :       reg ({})".format(
                            t, trainer.modifier, trainer.lr, loss_str, reg_str))

                trainer.finish_cycle()

    writeScene(scene, os.path.join(expr_dir, 'final_scene.json'),
               os.path.join(expr_dir, 'final_cloud.ply'))


if __name__ == "__main__":
    opt = FilterOptions().parse()

    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)

    # Create ground truth
    scene = readScene(opt.source, device="cpu")
    if opt.cloud:
        points = readCloud(opt.cloud, device="cpu")
        points_coords, _, _ = normalize_point_batch(
            points[:, :3].unsqueeze(0), NCHW=False)
        points[:, :3] = points_coords.squeeze(0)*2
        scene.loadPoints(points)

    trainImageFilter(scene, benchmark=opt.benchmark)
