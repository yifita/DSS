import os
import torch
import numpy as np
from collections import OrderedDict
from ..core.renderer import createSplatter, SmapeLoss, NormalLengthLoss
from ..core.camera import CameraSampler, PinholeCamera
from ..core.scene import Scene
from ..cuda import rasterize_forward
from ..misc import imageFilters
from pytorch_points.utils.pc_utils import read_ply
from pytorch_points.network import operations
from .matrixConstruction import batchLookAt


def viewFromError(nCam, gtImage, predImage, predPoints, projPoints, splatter, offset=None):
    allPositions = torch.from_numpy(read_ply("example_data/pointclouds/sphere_300.ply", nCam)).to(device=splatter.camera.device)[:, :3].unsqueeze(0)
    device = splatter.camera.device
    focalLength = splatter.camera.focalLength
    width = splatter.camera.width
    height = splatter.camera.height
    sv = splatter.camera.sv

    offset = offset or splatter.camera.focalLength*0.5
    fromP = allPositions * offset

    diff = torch.sum((gtImage - predImage).abs(), dim=-1)
    diff = torch.nn.functional.avg_pool2d(diff.unsqueeze(0), 9, stride=4, padding=4, ceil_mode=False, count_include_pad=False).squeeze(0)
    w = diff.argmax() % diff.shape[0]
    h = diff.argmax() // diff.shape[0]
    w *= 4
    h *= 4
    # average points projected inside this region
    _, knn_idx, _ = operations.group_knn(5, torch.tensor([w, h, 1], dtype=projPoints.dtype, device=projPoints.device).view(1, 1, 3).expand(projPoints.shape[0], -1, -1),
                                         projPoints, unique=False, NCHW=False)
    # B, 1, K
    PN = predPoints.shape[0]
    knn_points = torch.gather(predPoints.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, predPoints.shape[-1]))
    center = torch.mean(knn_points, dim=-2).to(device=device)
    ups = torch.tensor([0, 0, 1], dtype=center.dtype, device=device).view(1, 1, 3).expand_as(fromP)
    ups = ups + torch.randn_like(ups) * 0.0001
    rotation, position = batchLookAt(fromP, center, ups)
    cameras = []
    for i in range(nCam):
        cam = PinholeCamera(device=device, focalLength=focalLength, width=width, height=height, sv=sv)
        cam.rotation = rotation[:, i, :, :]
        cam.position = position[:, i, :]
        cameras.append(cam)

    return diff.max(), cameras


def removeOutlier(gtImage, projPoints, sigma=50):
    """
    treat gt image as 2D points, compare nn distance between projPoints with gt points
    return inlier confidence
    input:
        gtImages list of (1, H, W, C)
        projPoints (1, N, 2or3)
    """
    gtImage = gtImage.to(device=projPoints.device)
    _, H, W, _ = gtImage.shape
    # is projPoint inside silhouette?
    mask = torch.sum(gtImage.abs(), dim=-1) > 0
    # 1, N, 2
    gtPoints = torch.nonzero(mask).to(dtype=projPoints.dtype, device=projPoints.device)[None, :, 1:]
    # yx to xy
    gtPoints = torch.flip(gtPoints, [-1])
    # knn_d (1, N, 1)
    knn_p, knn_idx, knn_d = operations.faiss_knn(1, projPoints[:, :, :2], gtPoints, NCHW=False)
    projIJ = projPoints[0, :, :2].long()
    projIJ = torch.where(torch.any((projIJ >= H) | (projIJ < 0), dim=-1, keepdim=True), torch.full_like(projIJ, -1), projIJ)
    ijIndices = projIJ[:, 0] + projIJ[:, 1] * W
    # N, use gather_maps, reshape to BxHxWxk, ignore points outside image
    isInside = rasterize_forward.gather_maps(mask.view(1, -1, 1).cuda(), ijIndices.view(1, -1, 1, 1).cuda(), 1.0).view(-1).to(device=knn_d.device)
    # if point is inside, set knn_d to zero
    # N
    knn_d = knn_d.squeeze()
    knn_d = torch.where(isInside, torch.zeros_like(knn_d), knn_d)
    # confidence = sigmoid(knn_d)
    score = torch.exp(-knn_d/sigma/2)
    return score


def renderScene(refScene, opt, cameras):
    splatter = createSplatter(opt, refScene)
    splatter.setCloud(refScene.cloud)
    splatter.initCameras(cameras=cameras)

    groundtruth = []
    for i, cam in enumerate(refScene.cameras):
        splatter.setCamera(i)
        result = splatter.render().detach()
        groundtruth.append(result)
    return groundtruth


class Trainer(object):
    def __init__(self, opt, scene=None):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.device = opt.device
        self.save_dir = os.path.join(opt.output, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.visual_names = []
        self.learningRates = OrderedDict([_ for _ in zip(self.opt.modifiers, self.opt.learningRates)])
        self.optimizers = OrderedDict([(modifier, None) for modifier in opt.modifiers])
        self.metric = OrderedDict([(modifier, 0.0) for modifier in opt.modifiers])  # used for learning rate policy 'plateau'
        self.steps = OrderedDict([(modifier, step) for modifier, step in zip(opt.modifiers, np.cumsum(np.array(opt.steps, dtype="int32")))])
        self.model = createSplatter(opt, scene=scene)
        self.step = opt.startingStep
        self.imageLoss = SmapeLoss()
        self.normalLengthLoss = NormalLengthLoss()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def initialize_views(self):
        """Create new camera position set to self.cameras"""
        self.cameras = []
        for i in range(self.opt.genCamera):
            cam = next(self.camSampler)
            self.cameras.append(cam)

    def create_reference(self, refScene, cameras=None):
        """create new views, references, set groundtruths, allSunColors and allSunDirections"""
        if cameras is None and self.opt.genCamera > 0:
            self.initialize_views()
        elif cameras is None:
            self.cameras = refScene.cameras
        else:
            self.cameras = cameras
        with torch.no_grad():
            self.groundtruths = renderScene(refScene, self.opt, self.cameras)
            self.model.initCameras(self.cameras)
            self.model.allSunColors = refScene.allSunColors
            self.model.allSunDirections = refScene.allSunDirections

    def initiate_cycle(self):
        """reset scheduler"""
        for key, scheduler in self.schedulers.items():
            scheduler._reset()

    def finish_cycle(self):
        self.model.updateLocalSize(self.opt.backwardLocalSizeDecay)

    def forward(self, camID=None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not hasattr(self, "predictions"):
            self.predictions = [None] * len(self.cameras)
        if camID is None:
            for i, cam in enumerate(self.cameras):
                self.forward(i)
        else:
            self.camID = camID
            self.model.setCamera(camID)
            self.predictions[camID] = self.model.forward()

    def optimize_parameters(self):
        """Calculate losses, gradients, update parameters, apply regularization; called in every training iteration"""
        # set current modifier
        for key, step in self.steps.items():
            if (self.step % sum(self.opt.steps)) < step:
                self.modifier = key
                break
            else:
                continue
        for m in self.opt.modifiers:
            if m != self.modifier:
                getattr(self.model, m).requires_grad = False
            else:
                getattr(self.model, m).requires_grad = True

        # initialize before optimization
        self.loss_image = [0.0] * len(self.cameras)
        self.loss_reg = [0.0] * len(self.cameras)
        self.metric[self.modifier] = 0
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()
        self.model.clearVisibility()
        self.apply_projection = (self.opt.projectionFreq > 0 and (self.step+1) % self.opt.projectionFreq == 0)
        self.apply_repulsion = (self.opt.repulsionFreq > 0 and (self.step+1) % self.opt.repulsionFreq == 0)
        # compute gradient for each camera view
        nValidViews = 0
        for camID, cam in enumerate(self.cameras):
            if self.groundtruths[camID] is None:
                continue
            self.forward(camID)
            if self.predictions[camID] is None:
                continue
            loss = self.imageLoss(self.predictions[camID], self.groundtruths[camID].detach()) * self.opt.imageWeight
            self.metric[self.modifier] += loss.cpu().item()
            self.loss_image[camID] = loss.detach()
            # regularizer normal length
            if self.model.localNormals.requires_grad:
                reg = 0.001 * self.normalLengthLoss(self.model._localNormals).cuda()
                loss = loss + reg
                self.loss_reg[camID] = reg.cpu().detach().item()
            # regularizer repulsion
            if self.model.localPoints.requires_grad:
                if self.apply_repulsion:
                    occlusionCount = self.model.nonvisibility/(self.model.renderTimes.to(device=self.model.nonvisibility.device)+0.01)
                    reg = self.model.pointRegularizerLoss(self.model.cameraPoints,
                                                          self.model.localNormals.detach(),
                                                          occlusionCount,
                                                          self.model.renderable_indices,
                                                          use_density=True, include_projection=False)
                    if isinstance(reg, torch.Tensor):
                        reg = reg.cuda()
                        loss = loss + reg
                        self.loss_reg[camID] = reg.cpu().detach().item()
            loss.backward()
            nValidViews += 1

        # metric is the average over all cameras
        self.metric[self.modifier] /= nValidViews
        # average gradient for all views
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad /= (self.model.renderTimes + 1e-2)
        self.step += 1
        # clip gradients
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.clipGrad)
        self.optimizers[self.modifier].step()
        self.update_learning_rate()

        # projection
        if self.apply_projection:
            occlusionCount = self.model.nonvisibility/(self.model.renderTimes.to(device=self.model.nonvisibility.device)+0.01)
            renderable_list = (self.model.renderTimes > 0).nonzero()[:, 1].view(1, -1, 1).to(self.model.localNormals.device)
            self.model.applyProjection(self.model.localPoints.data, self.model.localNormals.data,
                                       occlusionCount, renderable_list)

    def setup(self, opt, cloud):
        """initialize point cloud, set modifier, create schedulers

        Parameters:
            cloud (Cloud class) -- initial point cloud
        """
        self.model.setCloud(cloud)
        for modifier in self.opt.modifiers:
            self.model.setModifier(modifier)

        if not (self.model.cameraPosition.requires_grad or self.model.cameraRotation.requires_grad) and self.opt.genCamera > 0:
            self.camSampler = CameraSampler(self.opt.genCamera*self.opt.cycles, self.opt.camOffset, self.opt.camFocalLength,
                                            points=self.model.localPoints, camWidth=self.opt.width, camHeight=self.opt.height,
                                            filename=self.opt.cameraFile)

        self.optimizers = OrderedDict([(modifier, torch.optim.SGD([getattr(self.model, modifier)], lr=lr, momentum=0.1, nesterov=True))
                                       for modifier, lr in self.learningRates.items()])
        self.schedulers = OrderedDict([(modifier, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True,
                                                                                             threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.001))
                                       for modifier, optimizer in self.optimizers.items()])

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self):
        """Update learning rates for all modifiers"""
        if isinstance(self.schedulers[self.modifier], torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.schedulers[self.modifier].step(self.metric[self.modifier])
        else:
            self.schedulers[self.modifier].step()

        self.lr = self.optimizers[self.modifier].param_groups[0]['lr']

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


class FilterTrainer(Trainer):
    def __init__(self, opt, scene=None):
        opt.modifiers = ["localNormals", "localPoints"]
        if opt.im_filter == "Pix2PixDenoising":
            opt.shading = "diffuse"
            opt.average_term = True
        self.filter_func = getattr(imageFilters, opt.im_filter)
        super().__init__(opt, scene=scene)

    def setup(self, opt, cloud):
        self.model.setCloud(cloud)
        for modifier in self.opt.modifiers:
            self.model.setModifier(modifier)

        if not (self.model.cameraPosition.requires_grad or self.model.cameraRotation.requires_grad) and self.opt.genCamera > 0:
            self.camSampler = CameraSampler(self.opt.genCamera*self.opt.cycles, self.opt.camOffset, self.opt.camFocalLength,
                                            points=self.model.localPoints, normals=self.model.localNormals,
                                            camWidth=self.opt.width, camHeight=self.opt.height,
                                            filename=None)

        self.optimizers = OrderedDict([(modifier, torch.optim.SGD([getattr(self.model, modifier)], lr=lr, momentum=0.1, nesterov=True))
                                       for modifier, lr in self.learningRates.items()])
        self.schedulers = OrderedDict([(modifier, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True,
                                                                                             threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.001))
                                       for modifier, optimizer in self.optimizers.items()])
        self.initial_parameters = OrderedDict([(modifier, getattr(self.model, modifier).clone()) for modifier in self.opt.modifiers])

    def create_reference(self, refScene, cameras=None):
        """create views, render, filter."""
        if self.opt.genCamera < 1:
            return
        self.initialize_views()
        self.model.initCameras(self.cameras)

        with torch.no_grad():
            if not self.opt.recursiveFiltering:
                model_parameters = OrderedDict([(modifier, getattr(self.model, modifier)) for modifier in self.opt.modifiers])
                current_params = OrderedDict([(modifier, p.clone()) for modifier, p in model_parameters.items()])
                for p, p_gt in zip(model_parameters.values(), self.initial_parameters.values()):
                    p.data.resize_as_(p_gt)
                    p.data.copy_(p_gt)

            self.forward()
            self.validCams = [i for i, p in enumerate(self.predictions) if p is not None]
            self.predictions = [self.predictions[i].detach()[0] for i in self.validCams]

            if self.filter_func.__name__ == "Pix2PixDenoising":
                self.groundtruths = self.filter_func(self.predictions, self.opt.pix2pix)
            else:
                self.groundtruths = self.filter_func(self.predictions)
            for i, pair in enumerate(zip(self.groundtruths, self.predictions)):
                post, pre = pair
                # prevent changing background
                self.groundtruths[i] = torch.where(pre == 0.0, torch.zeros_like(pre), post)
            if not self.opt.recursiveFiltering:
                # copy back parameter values
                for p, p_cur in zip(model_parameters.values(), current_params.values()):
                    p.data.resize_as_(p_cur)
                    p.data.copy_(p_cur)

    def optimize_parameters(self):
        # set current modifier
        for key, step in self.steps.items():
            if (self.step % sum(self.opt.steps)) < step:
                self.modifier = key
                break
            else:
                continue
        for m in self.opt.modifiers:
            if m != self.modifier:
                getattr(self.model, m).requires_grad = False
            else:
                getattr(self.model, m).requires_grad = True

        # initialize before optimization
        self.loss_image = [0.0] * len(self.cameras)
        self.loss_reg = [0.0] * len(self.cameras)
        self.metric[self.modifier] = 0
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()
        self.model.clearVisibility()
        self.apply_projection = (self.opt.projectionFreq > 0 and (self.step+1) % self.opt.projectionFreq == 0)
        self.apply_repulsion = (self.opt.repulsionFreq > 0 and (self.step+1) % self.opt.repulsionFreq == 0)
        # compute gradient for each camera view
        nValidViews = 0
        for camID in self.validCams:
            self.forward(camID)
            # learn normal from image
            if self.model.localNormals.requires_grad:
                loss = self.imageLoss(self.predictions[camID][0], self.groundtruths[camID].detach()) * self.opt.imageWeight
                self.metric[self.modifier] += loss.cpu().item()
                self.loss_image[camID] = loss.detach()
                reg = 0.001 * self.normalLengthLoss(self.model._localNormals).cuda()
                loss = loss + reg
                self.loss_reg[camID] = reg.cpu().detach().item()
                loss.backward()
            # points position update from regularization only
            elif self.model.localPoints.requires_grad:
                if self.apply_repulsion:
                    # occlusionCount = self.model.nonvisibility/(self.model.renderTimes.to(device=self.model.nonvisibility.device)+0.01)
                    reg = self.model.pointRegularizerLoss(self.model.cameraPoints,
                                                          self.model.localNormals.detach(),
                                                          torch.ones_like(self.model.nonvisibility),
                                                          self.model.renderable_indices,
                                                          use_density=False, include_projection=False)
                    reg = reg.cuda()
                    loss = reg
                    self.loss_reg[camID] = reg.cpu().detach().item()

                    loss.backward()
            nValidViews += 1

        # metric is the average over all cameras
        self.metric[self.modifier] /= nValidViews
        # average gradient for all views
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad /= (self.model.renderTimes + 1e-2)
        self.step += 1
        # clip gradients
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.clipGrad)
        self.optimizers[self.modifier].step()
        self.update_learning_rate()

        # projection
        if self.model.localPoints.requires_grad and self.apply_projection:
            renderable_list = (self.model.renderTimes > 0).nonzero()[:, 1].view(1, -1, 1).to(self.model.localNormals.device)
            occlusionCount = self.model.nonvisibility/(self.model.renderTimes.to(device=self.model.nonvisibility.device)+0.01)
            if self.opt.average_term:
                self.model.applyAverageTerm(self.model.localPoints.data, self.model.localNormals.data, self.initial_parameters["localPoints"], renderable_list)
            self.model.applyProjection(self.model.localPoints.data, self.model.localNormals.data,
                                       torch.ones_like(self.model.nonvisibility), renderable_list)
