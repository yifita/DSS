import torch
import os
import numpy as np
import torch.nn as nn
from pytorch_points.network import operations
from pytorch_points.utils.pc_utils import save_ply, read_ply
from ..utils.mathHelper import dot, div, mul, det22, normalize, mm, inverse22, inverse33
from ..utils.matrixConstruction import convertWorldToCameraTransform, batchLookAt, batchAffineMatrix
from ..cuda import rasterizeDSS, rasterizeRBF, guided_scatter_maps
from .scene import Scene


modifiers = ["localPoints",
             "pointColors",
             "localNormals",
             "cameraPosition",
             "cameraRotation",
             "pointPosition",
             "pointRotation",
             "pointScale",
             "pointlightPositions",
             "pointlightColors",
             "sunDirections",
             "sunColors",
             "ambientLight",
             ]

saved_variables = {}

def save_grad(name):
    def hook(grad):
        saved_variables[name] = grad.cpu()
    return hook

def _check_values(tensor):
    return not (torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)))

def _findEllipseBoundingBox(a, b, c, d):
    """
    Expects the parameters of the ellipse equation:
    a x ^ 2 + b y ^ 2 + c xy = d
    returns radius along x and y axis
    """
    # or: a x^2 + b y^2 + c xy = d # Eq 1
    # solve x on varying y
    # set determinant to zero -> y_max
    # same for x_max
    c2 = c**2
    ab4 = 4*a*b
    y = torch.sqrt(4*a*d/(ab4-c2))
    x = torch.sqrt(4*b*d/(ab4-c2))
    # x = (-c*y_/2/a).abs()
    # y = (-c*x_/2/a).abs()
    return (x, y)


def _genSunLights(camForward: torch.tensor, mode="triColor") -> torch.Tensor:
    """
    generate rgb sun lights depending on camera position
    """
    if mode == "triColor":
        # R around camera position
        rDir = normalize(camForward).cuda()
        rColor = torch.tensor([0.9, 0, 0], dtype=camForward.dtype).expand_as(rDir).cuda()

        bDir = normalize((rDir + torch.rand_like(rDir)).cross(rDir)).cuda()
        bColor = torch.tensor([0, 0.9, 0], dtype=camForward.dtype).expand_as(bDir).cuda()

        gDir = (bDir).cross(rDir).cuda()
        gColor = torch.tensor([0, 0.0, 0.9], dtype=camForward.dtype).expand_as(bDir).cuda()

        return torch.stack([rDir, bDir, gDir], dim=-2), torch.stack([rColor, bColor, gColor], dim=-2)
    else:
        Dir = normalize(camForward).cuda()
        color = torch.tensor([0.9, 0.9, 0.9], dtype=camForward.dtype).expand_as(Dir).cuda()
        return Dir.unsqueeze(-2), color.unsqueeze(-2)


def _computeDensity(points, knn_k=33, radius=0.1):
    radius2 = radius*radius
    if points.is_cuda:
        knn_points, knn_idx, distance2 = operations.group_knn(knn_k, points, points, unique=False, NCHW=False)
    else:
        knn_points, knn_idx, distance2 = operations.faiss_knn(knn_k, points, points, NCHW=False)
    knn_points = knn_points[:, :, 1:, :].contiguous().detach()
    knn_idx = knn_idx[:, :, 1:].contiguous()
    distance2 = distance2[:, :, 1:].contiguous()
    # ball query find center
    knn_points = torch.where(distance2.unsqueeze(-1)>radius2, torch.zeros_like(knn_points), knn_points)
    weight = torch.exp(-distance2/radius2/4)
    weight = torch.where(distance2>radius2, torch.zeros_like(weight), weight)
    weight = torch.sum(weight, dim=-1)
    return weight


def _findSplatBoundingBox(cutoffC, projPoint, IMAGE_WIDTH, IMAGE_HEIGHT, ellipseParams):
    """
    Compute the bounding box around each projected points given the elliptical parameters ax^2 + by^2 + cxy <= cutoffC
    input:
        cutoffC         scalar
        projPoint       BxNx2
        ellipseParams   BxNx3 coefficients a, b, c of the elliptical function
        image_width     scalar
        image_height    scalar
    """
    a = ellipseParams[:, :, 0]
    b = ellipseParams[:, :, 1]
    c = ellipseParams[:, :, 2]
    x = projPoint[:, :, 0]
    y = projPoint[:, :, 1]
    # BxN
    (xE, yE) = _findEllipseBoundingBox(a, b, c, cutoffC)
    if projPoint.requires_grad:
        # backward memory constraint
        xE = torch.min(xE, torch.full((1, 1), 15, dtype=xE.dtype, device=xE.device))
        yE = torch.min(yE, torch.full((1, 1), 15, dtype=xE.dtype, device=xE.device))
    xmin = x - xE
    xmax = x + xE + 1
    ymin = y - yE
    ymax = y + yE + 1
    ixmin = xmin.floor_().int()  # .clamp_(0, IMAGE_WIDTH - 1)
    ixmax = xmax.floor_().int()  # .clamp_(1, IMAGE_WIDTH)
    iymin = ymin.floor_().int()  # .clamp_(0, IMAGE_HEIGHT - 1)
    iymax = ymax.floor_().int()  # .clamp_(1, IMAGE_HEIGHT)
    result = torch.stack((ixmin, iymin, ixmax, iymax), -1)
    return result

class NormalLengthLoss(nn.Module):
    """enforce normal length to be 1"""
    def __init__(self):
        super(NormalLengthLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def forward(self, normals):
        squaredNorm = torch.sum(normals**2, dim=-1)
        return self.criterion(squaredNorm, torch.ones_like(squaredNorm))

class SmapeLoss(nn.Module):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """
    def __init__(self):
        super(SmapeLoss, self).__init__()

    def forward(self, x, y):
        """
        x pred  (N,3)
        y label (N,3)
        """
        return torch.mean(torch.abs(x-y)/(torch.abs(x)+torch.abs(y)+1e-2))

def createSplatter(opt, scene=None):
    if opt.type == "DSS":
        return DSS(opt, scene)
    elif opt.type == "Baseline":
        return Baseline(opt, scene)

class DSS(torch.nn.Module):
    def __init__(self, opt, scene=None):
        """
        The renderer is allowed to write to new fields into the scene (for caching purposes).
        But it does not change any input fields.
        """
        super(DSS, self).__init__()
        if scene is None:
            scene = Scene()
        self.lowPassBandWidth = 1  # in screen space
        self.mergeTopK = opt.mergeTopK
        self.considerZ = opt.considerZ
        # if the distance between two splats is less than this theshold, their values are merged (with 'mix' merge_strategy)
        self.merge_threshold = opt.mergeThreshold
        self.repulsion_radius = opt.repulsionRadius
        self.projection_radius = opt.projectionRadius
        self.repulsion_weight = opt.repulsionWeight
        self.projection_weight = opt.projectionWeight
        self.average_weight = opt.averageWeight
        self.sharpness_sigma = opt.sharpnessSigma
        self.cutOffThreshold = opt.cutOffThreshold
        self.Vrk_h = opt.Vrk_h
        self.backwardLocalSize = opt.backwardLocalSize
        self.shading = scene.cloud.shading
        self.device = opt.device
        if self.device is None:
            self.device = torch.device("cpu")
        self.minBackwardLocalSize = 32
        self.backfaceCulling = True

        # parameters
        self.pointlightPositions = nn.Parameter(scene.pointlightPositions, requires_grad=False).cuda()
        self.pointlightColors = nn.Parameter(scene.pointlightColors, requires_grad=False).cuda()
        self.sunDirections = nn.Parameter(scene.sunDirections, requires_grad=False).cuda()
        self.sunColors = nn.Parameter(scene.sunColors, requires_grad=False).cuda()
        self.ambientLight = nn.Parameter(scene.ambientLight.unsqueeze(0), requires_grad=False).cuda()
        self.cameraRotation = nn.Parameter(scene.cameras[0].rotation, requires_grad=False)
        self.cameraPosition = nn.Parameter(scene.cameras[0].position, requires_grad=False)
        self.scene = scene
        self.cameraInitialized = False
        self.cloudInitialized = False
        # N, 1
        self.register_buffer("nonvisibility", torch.zeros(1, 1).cuda())
        self.register_buffer("renderTimes", torch.zeros(1, 1))

    def initCameras(self, cameras, genSunMode="triColor"):
        self.cameras = self.scene.cameras = cameras
        # C, B, 3, 3
        self.allCameraRotation = torch.stack([c.rotation for c in self.cameras], dim=0)
        self.allCameraPosition = torch.stack([c.position for c in self.cameras], dim=0)
        # self.w2cs = torch.stack([c.world2CameraMatrix() for c in self.cameras], dim=0).to(device=self.cameraPosition.device)
        self.cameraInitialized = True
        sunDirs, sunColors = _genSunLights(self.allCameraRotation[:, :, :, 2], mode=genSunMode)
        self.allSunDirections = self.scene.allSunDirections = sunDirs
        self.allSunColors = self.scene.allSunColors = sunColors

    def setCamera(self, camID, genSun=True):
        self.camera = self.cameras[camID]
        if genSun:
            # sunDirs, sunColors = _genSunLights(self.camera.rotation[:, :, 2])
            self.sunDirections.set_(self.allSunDirections[camID,...])
            self.sunColors.set_(self.allSunColors[camID,...])
        self.cameraRotation.set_(self.allCameraRotation[camID,...])
        self.cameraPosition.set_(self.allCameraPosition[camID,...])
        self.w2c = self.camera.world2CameraMatrix(self.cameraRotation, self.cameraPosition)

    def setCloud(self, cloud):
        self.scene.cloud = cloud
        if self.cloudInitialized:
            self.localPoints.set_(cloud.localPoints.unsqueeze(0))
            self.pointColors.set_(cloud.color.unsqueeze(0))
            self.localNormals.set_(cloud.localNormals.unsqueeze(0))
            self.pointPosition.set_(cloud.position.unsqueeze(0))
            self.pointRotation.set_(cloud.rotation.unsqueeze(0))
            self.pointScale.set_(cloud.scale.unsqueeze(0))
            pShape = list(self.localPoints.shape)
            pShape[-1] = 1
            self.nonvisibility.resize_(*pShape).zero_()
            self.renderTimes.resize_(*pShape).zero_()
            self.renderTimes = self.renderTimes.to(device=self.localPoints.device)
            self.cloudInitialized = True
            # Create model to world matrix (4x4)
            if not hasattr(self, "m2w") or self.m2w.requires_grad:
                self.m2w = batchAffineMatrix(self.pointRotation, self.pointPosition, self.pointScale)
            return
        self.localPoints = nn.Parameter(cloud.localPoints.unsqueeze(0), requires_grad=False)
        self.pointColors = nn.Parameter(cloud.color.unsqueeze(0), requires_grad=False)
        self.localNormals = nn.Parameter(cloud.localNormals.unsqueeze(0), requires_grad=False)
        self.pointPosition = nn.Parameter(cloud.position.unsqueeze(0), requires_grad=False)
        self.pointRotation = nn.Parameter(cloud.rotation.unsqueeze(0), requires_grad=False)
        self.pointScale = nn.Parameter(cloud.scale.unsqueeze(0), requires_grad=False)
        pShape = list(self.localPoints.shape)
        pShape[-1] = 1
        self.m2w = batchAffineMatrix(self.pointRotation, self.pointPosition, self.pointScale)
        self.nonvisibility.resize_(*pShape).zero_()
        self.renderTimes.resize_(*pShape).zero_()
        self.renderTimes = self.renderTimes.to(device=self.localPoints.device)
        self.cloudInitialized = True

    def setModifier(self, modifierNames):
        for name, p in self.named_parameters():
            if name in modifierNames:
                p.requires_grad = True

    def pointRegularizerLoss(self, points_data, normals_data, nonvisibility_data, idxList=None, include_projection=False, use_density=False):
        if self.repulsion_weight <= 0 and self.projection_weight <= 0:
            return
        batchSize, PN, _ = points_data.shape
        if PN <= 3:
            return
        knn_k = 33
        normals_data = normalize(normals_data)
        points = points_data
        normals = normals_data
        nonvisibility_data = nonvisibility_data.to(device=points.device)
        nonvisibility = nonvisibility_data
        if idxList is not None:
            points = torch.gather(points, 1, idxList.expand(-1, -1, points.shape[-1]))
            nonvisibility = torch.gather(nonvisibility, 1, idxList.expand(-1, -1, nonvisibility.shape[-1]))
            normals = torch.gather(normals, 1, idxList.expand(-1, -1, normals.shape[-1]))

        PN = points.shape[1]
        rradius = self.repulsion_radius
        rradius2 = rradius**2
        # repulsion force to projPoints/cameraPoints
        iradius = 1/(rradius2)/2
        # first KNN (B, N, k, c)
        if points.is_cuda:
            knn_points, knn_idx, distance2 = operations.group_knn(knn_k, points, points_data, unique=False, NCHW=False)
        else:
            knn_points, knn_idx, distance2 = operations.faiss_knn(knn_k, points, points_data, NCHW=False)
            # distance2 = distance2 * distance2
        knn_points = knn_points[:, :, 1:, :].contiguous().detach()
        knn_idx = knn_idx[:, :, 1:].contiguous()
        distance2 = distance2[:, :, 1:].contiguous()
        knn_normals = torch.gather(normals_data.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, normals.shape[-1]))
        knn_v = knn_points - points.unsqueeze(dim=2)
        # phi, psi and theta are used for finding local plane
        # while only psi is used for repulsion loss weight
        # B, N, k
        phi = torch.gather(nonvisibility_data.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1)).squeeze(-1)
        # visibility = 1 / (nonvisibility+1)
        phi = 1/(phi+1)
        # # quantize phi, either 1 or 0.1
        # phi = torch.where(phi > 0, torch.full([1, 1, 1], 1.0, device=phi.device), torch.full([1, 1, 1], 0.5, device=phi.device))
        psi = torch.exp(-distance2*iradius)
        sharpness_bandwidth = max(1e-5, 1-np.cos(self.sharpness_sigma*180.0/3.1415926, dtype=np.float32))
        sharpness_bandwidth *= sharpness_bandwidth
        # B, N, k
        theta = torch.exp(-torch.pow(1-torch.sum(normals.unsqueeze(2)*knn_normals, dim=-1), 2)/sharpness_bandwidth)
        weight = phi*psi*theta
        weightSum = torch.sum(weight, dim=2, keepdim=True)
        weight /= (weightSum+1e-10)
        # project to local plane
        var = weight.unsqueeze(-1)*(knn_points - torch.sum(weight.unsqueeze(-1)*knn_points, dim=2, keepdim=True))
        # the previous step introduces small numeric error due to weighting
        var = torch.where(var.abs() / torch.max(var.abs(), dim=-1, keepdim=True)[0] < 1e-2, torch.zeros_like(var), var)
        # BN, k, 3
        _, _, V = operations.batch_svd(var.view(-1, knn_k-1, 3))
        V = V.detach()
        totalLoss = 0
        ploss = 0
        rloss = 0
        if include_projection and self.projection_weight > 0:
            # projection minimize distance to the plane
            Vp = V.clone()
            # BN, k, 3, 1
            Vn = Vp.unsqueeze(1)[:, :, :, 2:3]
            # BN, k, 3
            knn_v_p = knn_v.clone()
            # x@V@Vt
            projection_v = torch.matmul(torch.matmul(knn_v_p.view(-1, knn_k-1, 1, 3), Vn), Vn.transpose(-2,-1)).squeeze(-2)
            # BN, k
            distance2 = torch.sum(projection_v*projection_v, dim=-1)
            # B,N,k
            distance2 = distance2.view(batchSize, -1, knn_k-1)
            # weight with visibility and angular, distance similarity
            ploss = torch.mean(distance2*weight.detach())*self.projection_weight
            loss = torch.where(distance2 > rradius2, torch.zeros_like(ploss), ploss)
            totalLoss += ploss

        if self.repulsion_weight > 0:
            # repulsion proj to the first two principle axes, set last column of V to zero
            # BN, 3, 3
            V[:, :, -1] = 0
            # BN, k, 1, 3
            V = V.unsqueeze(-3).expand(-1, knn_k-1, -1, -1)
            # BN, k, 3
            knn_v_r = knn_v.clone()
            knn_v_r.register_hook(lambda x: x.clamp(-0.02, 0.02))
            # BN, k, 3
            repulsion_v = torch.matmul(torch.matmul(knn_v_r.view(-1, knn_k-1, 1, 3), V), V.transpose(-2, -1)).squeeze(-2)
            # repulsion_v = knn_v_r
            # BN, k
            distance2 = torch.sum(repulsion_v * repulsion_v, dim=-1)
            distance2 = distance2.view(batchSize, -1, knn_k-1)
            # loss = torch.exp(-distance2*iradius)
            rloss = 1/torch.sqrt(distance2+1e-4)
            # loss = -distance2
            # loss = 1/(distance2+0.001)
            # (torch.sqrt(distance2+1e-8) - self.repulsion_radius)**2
            rloss = torch.where(distance2 > rradius2, torch.zeros_like(rloss), rloss)
            # B,N,k
            weight = torch.where(distance2 > rradius2, torch.zeros_like(psi), psi)
            if use_density:
                densityWeights = _computeDensity(points)
                weight = weight * densityWeights.unsqueeze(-1)
            weightSum = torch.sum(weight, dim=-1, keepdim=True)+1e-8
            rloss = rloss * weight.detach()
            # B,N
            rloss /= weightSum
            rloss = torch.mean(rloss)*self.repulsion_weight
            totalLoss += rloss

        if include_projection:
            return ploss, rloss
        return totalLoss

    def applyAverageTerm(self, points_data, normals_data, original_points, idxList=None, original_density=None):
        """
        points          B,N,3
        original_points B,N,3
        original_density B,N,1
        """
        points = points_data
        if idxList is not None:
            points = torch.gather(points_data, 1, idxList.expand(-1, -1, points_data.shape[-1]))
            normals = torch.gather(normals_data, 1, idxList.expand(-1, -1, normals_data.shape[-1]))

        PN = points.shape[1]

        knn_k = 16
        if points.is_cuda:
            knn_points, knn_idx, distance2 = operations.group_knn(knn_k, points, original_points, unique=False, NCHW=False)
        else:
            knn_points, knn_idx, distance2 = operations.faiss_knn(knn_k, points, original_points, NCHW=False)
        radius2 = self.repulsion_radius*self.repulsion_radius
        # ball query find center
        knn_points = torch.where(distance2.unsqueeze(-1)>radius2, torch.zeros_like(knn_points), knn_points)
        weight = torch.exp(-distance2/radius2/4)
        weight = torch.where(distance2>radius2, torch.zeros_like(weight), weight)
        # original density term
        if original_density is not None:
            if original_density.dim() == 3:
                original_density = original_density.squeeze(-1)
            original_density_weight = torch.gather(original_density.unsqueeze(1).expand(-1, PN, -1), 2, knn_idx)
            original_density_weight = torch.where(distance2>radius2, torch.zeros_like(original_density_weight), original_density_weight)
            weight = weight * original_density_weight

        weightSum = torch.sum(weight, dim=-1, keepdim=True) + 1e-8
        weight /= weightSum

        # find average
        originalAverage = torch.sum(knn_points * weight.unsqueeze(-1), dim=-2)
        # project to its normal
        update = dot(originalAverage - points, normals, dim=-1).unsqueeze(-1) * normals * self.average_weight
        if idxList is not None:
            points_data.scatter_add_(1, idxList.expand(-1, -1, points_data.shape[-1]), update)
            return
        points += update


    def applyProjection(self, points_data, normals_data, nonvisibility_data, idxList=None, decay=1.0):
        if self.projection_weight <= 0:
            return
        batchSize, PN, _ = points_data.shape
        if PN <= 3:
            return
        normals_data = normalize(normals_data)
        knn_k = 33
        sharpness_sigma = self.sharpness_sigma
        projection_weight = self.projection_weight
        rradius = self.projection_radius
        points = points_data
        normals = normals_data
        nonvisibility_data = nonvisibility_data.to(device=points.device)
        nonvisibility = nonvisibility_data
        if idxList is not None:
            points = torch.gather(points, 1, idxList.expand(-1, -1, points.shape[-1]))
            normals = torch.gather(normals, 1, idxList.expand(-1, -1, normals.shape[-1]))
            nonvisibility = torch.gather(nonvisibility, 1, idxList.expand(-1, -1, nonvisibility.shape[-1]))
        PN = points.shape[1]
        rradius2 = rradius**2
        # repulsion force to projPoints/cameraPoints
        iradius = 1/(rradius2)/4
        # first KNN (B, N, k, c)
        if points.is_cuda:
            knn_points, knn_idx, distance2 = operations.group_knn(knn_k, points, points_data, unique=False, NCHW=False)
        else:
            knn_points, knn_idx, distance2 = operations.faiss_knn(knn_k, points, points_data, NCHW=False)
            # distance2 = distance2 * distance2
        knn_points = knn_points[:, :, 1:, :].contiguous()
        knn_idx = knn_idx[:, :, 1:].contiguous()
        distance2 = distance2[:, :, 1:].contiguous()
        if torch.all(distance2[:, :, 0] > rradius2):
            return
        knn_normals = torch.gather(normals_data.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, normals.shape[-1]))

        # give invisible points a small weight
        phi = torch.gather(nonvisibility_data.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1)).squeeze(-1)
        # phi = torch.where(phi > 0, torch.full([1, 1, 1], 1.0), torch.full([1, 1, 1], 1.0))
        phi = 1 / (1+phi)

        # B, N, k
        theta = torch.exp(-distance2*iradius)
        # B, N, k
        sharpness_bandwidth = max(1e-5, 1-np.cos(sharpness_sigma*180.0/3.1415926, dtype=np.float32))
        sharpness_bandwidth *= sharpness_bandwidth
        # B, N, k
        psi = torch.exp(-torch.pow(1-torch.sum(normals.unsqueeze(2)*knn_normals, dim=-1), 2)/sharpness_bandwidth)
        weight = psi * theta * phi
        weight = torch.where(distance2 > rradius2, torch.zeros_like(weight), weight)
        # B, N, k, dot product
        project_dist_sum = torch.sum((points.unsqueeze(2) - knn_points)*knn_normals, dim=-1)*weight
        # B, N, 1
        project_dist_sum = torch.sum(project_dist_sum, dim=-1, keepdim=True)+1e-10
        # B, N, 1
        project_weight_sum = torch.sum(weight, dim=-1, keepdim=True)+1e-10
        # B, N, c
        normal_sum = torch.sum(knn_normals*weight.unsqueeze(-1), dim=2)

        update_normal = normal_sum/project_weight_sum
        update_normal = normalize(update_normal)
        # too few neighbors or project_weight_sum too small
        update_normal = torch.where((torch.sum(distance2 <= rradius2, dim=-1) < 3).unsqueeze(-1) | (project_weight_sum < 1e-7), torch.zeros_like(update_normal), update_normal)
        point_update = -(update_normal * (project_dist_sum / project_weight_sum))
        point_update *= (self.projection_weight*decay)
        point_update = torch.clamp(point_update, -0.01, 0.01)
        if not _check_values(point_update):
            import pdb; pdb.set_trace()
        # apply this force
        if idxList is not None:
            points_data.scatter_add_(1, idxList.expand(-1, -1, points_data.shape[-1]), point_update)
            return
        points_data += point_update
        if self.verbose:
            saved_variables["projection"] = point_update.cpu()
            saved_variables["pweight"] = weight.cpu()

    def world2CameraMatrix(self, rotation, position):
        """
        4x4 view matrix: P = K[R|t]
        """
        P = torch.eye(4, dtype=rotation.dtype).to(device=rotation.device)
        (R, t) = convertWorldToCameraTransform(rotation, position)
        P[:3, :3] = R
        P[:3, -1] = t
        return P

    def computeVr(self, cameraPoints):
        """
        cameraPoints  BxNx3or4
        Vrk per point BxNx2x2
        """
        # V_k^r: variance matrices of the basis functions r_k
        h = self.Vrk_h
        Vr = torch.zeros([cameraPoints.size(0), cameraPoints.size(1), 2, 2], device=cameraPoints.device)
        if self.scene.cloud.VrkMode == "constant":
            # for simplicity, let V be constant
            h = h*h
        elif self.scene.cloud.VrkMode == "nearestNeighbor":
            # use ball query
            pts = cameraPoints[:, :, 0:3].detach().contiguous()
            # BxPx6
            _, _, distance = operations.faiss_knn(6, pts, pts, NCHW=False)
            h = torch.mean(distance, dim=2)
            h = h*h
        else:
            print("unknown VrkMode encountered: " + self.scene.cloud.VrkMode)
            h = (h*h)
        Vr[:, :, 0, 0] = h
        Vr[:, :, 1, 1] = h
        return Vr

    def pickRenderablePoints(self, normalAngle, cameraPoints):
        """
        points are renderable when
        1. they are in front of the camera (z > 0)
        2. they are pointing orthogonal to the viewing ray
        if backfaceCulling (default):
        3. their surface normals points towards the camera
        4. out of camera angle
        return:
            (X, 2) indice list
        """
        render_point = normalAngle.abs() > 0.000001
        render_point = render_point & (cameraPoints[:, :, 2] > 0)
        # if self.scene.cloud.backfaceCulling:
        render_point = render_point & (normalAngle >= 0.05)
        render_point = render_point & (torch.abs(cameraPoints[:, :, 0] / cameraPoints[:, :, 2]) < (self.camera.width/self.camera.focalLength/self.camera.sv))
        render_point = render_point & (torch.abs(cameraPoints[:, :, 1] / cameraPoints[:, :, 2]) < (self.camera.width/self.camera.focalLength/self.camera.sv))
        # X
        indices = torch.nonzero(render_point).detach()
        return indices

    def filterRenderablePoints(self):
        """
        target is only the renderable points
        return false if one example in the batch is not renderable
        """
        batchSize = self.cameraPoints.shape[0]
        indices = self.pickRenderablePoints(self.normalAngle, self.cameraPoints)
        numRenderables = torch.zeros((batchSize,), device=self.cameraPoints.device, dtype=torch.int64)
        for b in range(batchSize):
            numRenderables[b] = torch.sum(indices[:, 0] == b)

        if numRenderables.min() == 0:
            return False

        uniNumRenderables = numRenderables.max()
        # BxX, used for gather
        filledIndices = torch.zeros((batchSize, uniNumRenderables), device=indices.device, dtype=indices.dtype)
        accuNumRenderables = torch.cumsum(numRenderables, 0)
        for b in range(batchSize):
            filledIndices[b, :numRenderables[b]] = indices[accuNumRenderables[b]-numRenderables[b]:accuNumRenderables[b], 1]
            filledIndices[b, numRenderables[b]:] = filledIndices[b, numRenderables[b]-1]

        filledIndices = filledIndices.unsqueeze(-1)
        self.renderable_indices = filledIndices

        renderTimes = self.renderTimes.clone()
        renderTimes.zero_().scatter_(1, filledIndices.to(device=renderTimes.device), torch.ones_like(filledIndices, dtype=self.renderTimes.dtype, device=renderTimes.device))
        self.renderTimes  += renderTimes
        self._localPoints = torch.gather(self.localPoints, 1, filledIndices.expand(-1, -1, self.localPoints.shape[-1])).cuda()
        self._cameraPoints = torch.gather(self.cameraPoints, 1, filledIndices.expand(-1, -1, self.cameraPoints.shape[-1])).cuda()
        self._cameraNormals = torch.gather(self.cameraNormals, 1, filledIndices.expand(-1, -1, self.cameraNormals.shape[-1])).cuda()
        self._localNormals = torch.gather(self.localNormals, 1, filledIndices.expand(-1, -1, self.localNormals.shape[-1])).cuda()
        self._color = torch.gather(self.pointColors, 1, filledIndices.expand(-1, -1, self.pointColors.shape[-1])).cuda()
        return True

    def computeWk(self, mode, pointColors, cameraNormals, localNormals, ambientLight, cameraPoints,
                  cameraSuns, cameraPointlights):
        """
        apply albedo, normal, sun shading, point light shading to points
        input:
            pointColors   B x P x 3 point colors
            cameraNormals B x P x 3 point normals in camera space
            cameraSuns    B x S x 6 direction and color of sun
            cameraPointlights    B x S x 6 direction and color of point lights
            ambientLight  B x 3
        output:
            shade         B x P x 3
        """
        if mode == "albedo":
            return pointColors
        if mode == "depth":
            # rescale it to make a difference visual
            invdepths = 1/cameraPoints[:, :, 2].unsqueeze(-1).contiguous()
            invdepths = invdepths - torch.min(invdepths, dim=1, keepdim=True)[0]
            invdepths /= torch.max(invdepths, dim=1, keepdim=True)[0]
            return invdepths
        if mode == "normal":
            color = cameraNormals[:, :, :3]
            color = torch.stack([(cameraNormals[:,:,2]+1)/2, (cameraNormals[:,:,1]+1)/2, (cameraNormals[:,:,0]+1)/2], dim=-1)
            # color /= torch.max(color, dim=0, keepdim=True)[0]
            return color
        if mode == "diffuse":
            # ambient color B x 3
            shade = albedoMap = ambientLight.unsqueeze(-2) * pointColors
            # sun: color = MaterialDiffuseColor * LightColor * cosTheta;
            if cameraSuns is not None and cameraSuns.size(0) != 0:
                assert(cameraSuns.shape[2] == 6), "cameraSun must be a Sx6 tensor"
                sunDirs = -cameraSuns[:, :, :3]  # camera outgoing ray
                sunColors = cameraSuns[:, :, 3:]
                # BxSx3 @ Bx3xP = BxSxP
                cosAlpha = sunDirs.matmul(cameraNormals.transpose(1, 2))
                cosAlpha = torch.clamp(cosAlpha, 0.0, 1.0)
                # BxSxPx1 * BxSx1x3
                sunShade = torch.sum(cosAlpha.unsqueeze(-1) *
                                     sunColors.unsqueeze(2), dim=1) * pointColors
                shade += sunShade

            # point: same as above, but light dir is not uniform
            if cameraPointlights is not None and cameraPointlights.size(0) != 0:
                lightDir = cameraPointlights[:, :, :3]
                lightColor = cameraPointlights[:, :, 3:]
                # BxLx1x3 - Bx1xPx3
                lightDir = lightDir.unsqueeze(2) - cameraPoints[:, :, :3].unsqueeze(1)  # from point light to model
                lightDir = normalize(lightDir, -1)
                # BxLxPx3 * Bx1xPx3 -> BxLxPx1
                cosAlpha = torch.sum(lightDir * cameraNormals.unsqueeze(0), dim=-1, keepdim=True)
                cosAlpha = torch.clamp(cosAlpha, 0.0, 1.0)
                # BxLxPx1 * BxLx1x3
                pointlight = torch.sum(cosAlpha * lightColor.unsqueeze(2), dim=0) * pointColors
                shade += pointlight

            return shade

    def computeRho(self, projPoints, cameraPoints, cameraNormals, cutoffThreshold,
                   Vrk, width, height, camFar, lowPassBandWidth):
        """
            projPoints    BxNx2
            cameraPoints  BxNx3
            cameraPoints  BxNx3
        return:
            rho           BxNxbbHxbbW
            boundingBoxes BxNx4 xmin,ymin,xmax,ymax
            depthMap      BxNxbbHxbbWx3
        """
        if cameraPoints.dim() == 2:
            PN = cameraPoints.size()[0]
            cameraPoints = cameraPoints.unsqueeze(0)
        elif cameraPoints.dim() == 3:
            PN = cameraPoints.size()[1]
        else:
            raise ValueError("cameraPoints has wrong dimension")
        batchSize = cameraPoints.shape[0]
        # BxNx3
        u0, u1, x0plane, x1plane = self.computeUs(projPoints, cameraPoints, cameraNormals)
        # compute J = s_vp * J_pr * s_mv, Svp is absorbed in Jpr, Smv is absorbed in world2camera
        JprI = torch.zeros([batchSize, PN, 2, 2], device=cameraPoints.device)
        JprI[:, :, 0, 0] = dot(x0plane, u0, -1)
        JprI[:, :, 0, 1] = dot(x1plane, u0, -1)
        JprI[:, :, 1, 1] = dot(x1plane, u1, -1)
        Jprs = torch.inverse(JprI)
        Js = Jprs
        invJs = JprI
        invJsDet = det22(invJs.view(-1, 2, 2)).abs().view(batchSize, PN)

        # paper:
        # V^h: low pass filter
        # warped basis function: r_k'(x) = 1/|J^-1| G_{JV_k^r J^T}(x)
        # low-pass filter: h(x) = G_V^h(x)
        Vh = torch.eye(2, device=Js.device) * lowPassBandWidth
        Vh = Vh.reshape(1, 1, 2, 2).expand_as(JprI)
        # M matrix for cutoff
        # GV = J V_k^T J^T + I
        # BxPNx2x2
        GVs = Vh + torch.matmul(Js, torch.matmul(Vrk, Js.transpose(2, 3)))
        GVdets = det22(GVs.view(-1, 2, 2)).view(batchSize, PN)
        GVinvs = torch.inverse(GVs)
        Ms = 0.5*GVinvs
        # ellipseParams (a,b,c) = ax^2+cxy+by^2
        ellipseParams = torch.empty([batchSize, PN, 3], device=cameraPoints.device)
        ellipseParams[:, :, 0] = Ms[:, :, 0, 0]
        ellipseParams[:, :, 1] = Ms[:, :, 1, 1]
        ellipseParams[:, :, 2] = Ms[:, :, 0, 1] + Ms[:, :, 1, 0]
        # gaussian normalization term
        Gas = 1.0 / torch.sqrt(GVdets) / invJsDet / 2 / 3.1415926
        # BxNx4
        boundingBoxes = _findSplatBoundingBox(
            cutoffThreshold,
            projPoints[:, :, 0:2],
            width, height,
            ellipseParams).detach()
        width = torch.max(boundingBoxes[:, :, 2] - boundingBoxes[:, :, 0]).item()
        height = torch.max(boundingBoxes[:, :, 3] - boundingBoxes[:, :, 1]).item()

        # B x height x width
        ygrid, xgrid = torch.meshgrid(torch.arange(height, dtype=projPoints.dtype, device=projPoints.device),
                        torch.arange(width,  dtype=projPoints.dtype, device=projPoints.device))
        ygrid = ygrid.unsqueeze(0).expand(batchSize, -1, -1)
        xgrid = xgrid.unsqueeze(0).expand(batchSize, -1, -1)

        # B x N x height x width x 2
        pixs = torch.stack([xgrid, ygrid], dim=-1).unsqueeze(1).expand(-1, PN, -1, -1, -1)
        pixs = pixs + boundingBoxes[:, :, :2].unsqueeze(2).unsqueeze(2).to(dtype=pixs.dtype)

        # grid of camera-plane coordinates relative to projected point (B, N, H, W, 2)
        pixs = pixs - projPoints[:, :, :2].unsqueeze(2).unsqueeze(2)
        # B x N x H x W x 2 x 1
        pixs_ = pixs.unsqueeze(-1)
        # (B x N x H x W x 1 x 2) @ BxN x 1 x 1 x 2 x 2 @ (N x H x W x 2 x 1) -> BxNxHxW
        betas = pixs_.transpose(-2, -1).matmul(Ms.unsqueeze(2).unsqueeze(2)).matmul(pixs_).squeeze(-1).squeeze(-1)
        # BxN x H x W x 3
        inplane = pixs[:, :, :, :, 0].unsqueeze(-1) * x0plane.unsqueeze(2).unsqueeze(2) + pixs[:,:, :, :, 1].unsqueeze(-1) * x1plane.unsqueeze(2).unsqueeze(2) + \
            cameraPoints.unsqueeze(2).unsqueeze(2)[:, :, :, :3]
        # B x N x H x W
        depths = inplane[:, :, :, :, 2]
        # B x N x H x W
        Gbs = torch.exp(-betas)
        outofSupport = (betas > cutoffThreshold)
        Gbs = torch.where(outofSupport, torch.zeros_like(Gbs), Gbs)
        depths = torch.where(outofSupport, torch.full((1, 1, 1, 1), camFar, dtype=depths.dtype, device=depths.device), depths)
        inplane[:, :, :, :, 2] = depths
        # BxN x H x W
        rhos = Gas.unsqueeze(2).unsqueeze(2) * Gbs
        return rhos, Gas, boundingBoxes, inplane, Ms.contiguous()

    def computeUs(self, projPoints, cameraPoints, cameraNormals):
        """
        compute the basis vectors (u_0 and u_1) of the local parameterisation around each point
        input:
            projPoints (B, N, 3) homogen projected points
            cameraPoints (B, N, 3or4) homogen camera points
            cameraNormals (B, N, 3)  camera normals
        output:
            u0, u1, x0plane, x1plane (B, N, 3)
        """
        # We need to find u0 and u1 as described by "Surface Splatting, Zwicker et. al"
        # for that: project x0, x1 along camera direction onto tangent-plane: y_0, y_1
        # the paper defines u0 to be parallel to y_0: u0 = y_0 / ||y_0||
        # create x0 and x1 in projected space
        projX0 = projPoints.clone()
        projX1 = projPoints.clone()
        projX0[:, :, 0] += 1
        projX1[:, :, 1] += 1

        # (2, B, N, 3) back project shifted point to 3D
        x01cam = self.camera.backproject(torch.stack([projX0, projX1], dim=0), cameraPoints, cameraNormals)

        # the paper gives us u0 = y0 / ||y0|| (in plane), hence: u0 = x0n * x0ts - points to get x0InPlane, and normalize
        x01plane = x01cam - cameraPoints[:, :, :3]
        # B, N, 3
        x0plane, x1plane = torch.unbind(x01plane, dim=0)
        # normalize:
        u0 = normalize(x0plane, -1)
        u1 = u0.cross(cameraNormals, -1)
        return u0, u1, x0plane, x1plane

    def _need_to_compute(self, name):
        try:
            return getattr(self, name).requires_grad
        except AttributeError:  # this attribute not initialized yet
            return True
        else:
            return False

    def convertToCameraSpace(self):
        """
        convert localPoints and localNormals to camera space
        cameraPoints = m2c*localPoitns = w2c*m2w*localPoints
        cameraPoints / cameraPoints[:,3]
        """
        # localPoints (PN, 3)
        if self.localPoints.dim() == 2:
            PN = self.localPoints.size()[0]
        else:
            PN = self.localPoints.size()[1]
        # Create model to world matrix (4x4)
        if self._need_to_compute("m2w"):
            self.m2w = batchAffineMatrix(self.pointRotation, self.pointPosition, self.pointScale)
        # depending on camera model, gives the right world-to-camera matrix
        if self._need_to_compute("w2c"):
            self.w2c = self.world2CameraMatrix(self.cameraRotation, self.cameraPosition)

        self.m2c = torch.matmul(self.w2c, self.m2w)
        # self.Smv = self.pointScale
        # self.Svp = self.camera.sv
        # create 4d homogeneous points
        pShape = list(self.localPoints.shape)
        pShape[-1] = 1
        homPoints = torch.cat((self.localPoints, torch.ones(pShape, device=self.localPoints.device)), -1)
        # points in camera space
        self.cameraPoints = torch.matmul(homPoints, self.m2c.transpose(1, 2))[:, :, :3].contiguous()
        # transform the normals
        # self.worldNormals = torch.matmul(self.localNormals, self.m2w[:, :3, :3].transpose(1,2))
        self.cameraNormals = torch.matmul(self.localNormals, self.m2c[:, :3, :3].transpose(1,2))
        # normalize since m2w, m2c can have scaling scale
        # self.worldNormals = normalize(self.worldNormals, -1)
        self.cameraNormals = normalize(self.cameraNormals, -1)
        # from the point's perspective, where is the camera
        camDir = -normalize(self.cameraPoints, -1)
        self.normalAngle = dot(camDir, self.cameraNormals, -1)
        if not self.backfaceCulling:
            self.cameraNormals = torch.where(self.normalAngle.unsqueeze(-1) < 0, -self.cameraNormals, self.cameraNormals)
            self.normalAngle = dot(camDir, self.cameraNormals, -1)

        # transform light source to camera view
        if self.pointlightPositions is None or self.pointlightPositions.size()[0] == 0:
            self.cameraPointlights = None
        else:
            pShape = list(self.pointlightPositions.size())
            pShape[-1] = 1
            homLights = torch.cat((self.pointlightPositions, torch.ones(pShape, device=self.pointlightPositions.device)), dim=-1)
            self.cameraPointlights = torch.matmul(homLights, self.w2c.cuda().transpose(1, 2))
            self.cameraPointlights = torch.cat([self.cameraPointlights[:, :3], self.pointlightColors], dim=-1)

        if self.sunDirections is None or self.sunDirections.size()[0] == 0:
            self.cameraSuns = None
        else:
            self.cameraSuns = torch.matmul(self.sunDirections, self.w2c[:, :3, :3].cuda().transpose(1, 2))
            self.cameraSuns = normalize(self.cameraSuns, -1)
            self.cameraSuns = torch.cat([self.cameraSuns, self.sunColors], dim=-1)

    def updateLocalSize(self, decay):
        if self.backwardLocalSize is not None:
            self.backwardLocalSize *= decay
            self.backwardLocalSize = round(max(self.minBackwardLocalSize, self.backwardLocalSize))

    def render(self, **kwargs):
        assert(self.cloudInitialized), "Must call setCloud() before invoking render()"
        self.convertToCameraSpace()
        if not self.filterRenderablePoints():
            return None
        numPoint = self._cameraPoints.shape[1]
        if numPoint == 0:
            print("No renderable points")
            return None
        batchSize, numTotalPoints, _ = self.cameraPoints.shape
        # saved_variables["renderable_idx"] = self.renderable_indices.detach().cpu()
        # saved_variables["dIdp"] = torch.zeros([batchSize, numTotalPoints, 3], dtype=self.cameraPoints.dtype)
        # saved_variables["dIdpMap"] = torch.zeros((self.projPoints.shape[0], self.camera.height, self.camera.width, 2), dtype=self.projPoints)
        # saved_variables["projPoints"] = self.camera.projectPoints(self.cameraPoints)

        self._projPoints = self.camera.projectPoints(self._cameraPoints)
        Vr = self.computeVr(self._cameraPoints)

        result = self.computeRho(self._projPoints.detach(),
                                 self._cameraPoints.detach(),
                                 self._cameraNormals.detach(), self.cutOffThreshold,
                                 Vr.detach(), self.camera.width, self.camera.height,
                                 self.camera.far, self.lowPassBandWidth)
        # rho is the filter value at pixel x
        # rho is the filter value at ellipse center
        # ellipse bounding box
        # screen plane back-projected to 3D
        rho, rhoValues, boundingBoxes, inPlane, Ms = result
        Ws = self.computeWk(self.shading, self._color,
                            self._cameraNormals, self._localNormals, self.ambientLight, self._cameraPoints,
                            self.cameraSuns, self.cameraPointlights)
        final, pointIdxMap, rhoMap, WsMap, isBehind = rasterizeDSS(rho, rhoValues, Ws,
                          self._projPoints,
                          boundingBoxes,
                          inPlane, Ms,
                          self._cameraPoints[:, :, :3].contiguous(),
                          self.camera.width, self.camera.height,
                          self.camera.far, self.camera.focalLength,
                          localWidth=self.backwardLocalSize, localHeight=self.backwardLocalSize,
                          mergeThreshold=self.merge_threshold, considerZ=self.considerZ,
                          topK=self.mergeTopK)
        # compute occluded: isBehind = 1 and filterRho = 0
        occludedMap = (isBehind == 1) & (rhoMap == 0)
        self.local_occlusion = guided_scatter_maps(numPoint, occludedMap.unsqueeze(-1), pointIdxMap, boundingBoxes)
        self.nonvisibility.scatter_add_(1, self.renderable_indices.to(device=self.nonvisibility.device),
                                           self.local_occlusion.to(device=self.nonvisibility.device, dtype=self.nonvisibility.dtype))
        final = final.to(device=self._cameraPoints.device)
        return final

    def clearVisibility(self):
        self.nonvisibility.zero_()
        self.renderTimes.zero_()

    def forward(self):
        return self.render()


class Baseline(DSS):
    def __init__(self, opt, scene=None):
        """
        The renderer is allowed to write to new fields into the scene (for caching purposes).
        But it does not change any input fields.
        """
        super(Baseline, self).__init__(opt, scene)


    def render(self, **kwargs):
        assert(self.cloudInitialized), "Must call setCloud() before invoking render()"
        self.convertToCameraSpace()
        self.filterRenderablePoints()
        numPoint = self._cameraPoints.shape[1]
        if numPoint == 0:
            print("No renderable points")
            return None
        self._projPoints = self.camera.projectPoints(self._cameraPoints)

        Vr = self.computeVr(self._cameraPoints)
        result = self.computeRho(self._projPoints.detach(),
                                 self._cameraPoints.detach(),
                                 self._cameraNormals.detach(), self.cutOffThreshold,
                                 Vr.detach(), self.camera.width, self.camera.height,
                                 self.camera.far, self.lowPassBandWidth)
        # rho is the filter value at pixel x
        # rhoValues is the filter value at ellipse center
        # ellipse bounding box
        # screen plane back-projected to 3D
        rho, rhoValues, boundingBoxes, inPlane, Ms = result

        Ws = self.computeWk(self.shading, self._color,
                            self._cameraNormals, self._localNormals, self.ambientLight, self._cameraPoints,
                            self.cameraSuns, self.cameraPointlights)

        final, pointIdxMap, rhoMap, WsMap, isBehind = rasterizeRBF(rho, rhoValues, Ws,
                          self._projPoints,
                          boundingBoxes,
                          inPlane, Ms,
                          self._cameraPoints[:, :, :3].contiguous(),
                          self.camera.width, self.camera.height,
                          self.camera.far, self.camera.focalLength,
                          localWidth=self.backwardLocalSize, localHeight=self.backwardLocalSize,
                          mergeThreshold=self.merge_threshold, considerZ=self.considerZ,
                          topK=self.mergeTopK)
        # compute occluded: isBehind = 1 and filterRho = 0
        occludedMap = (isBehind == 1) & (rhoMap == 0)
        self.local_occlusion = guided_scatter_maps(numPoint, occludedMap.unsqueeze(-1), pointIdxMap, boundingBoxes)
        self.nonvisibility.scatter_add_(1, self.renderable_indices.to(device=self.nonvisibility.device),
                                           self.local_occlusion.to(device=self.nonvisibility.device, dtype=self.nonvisibility.dtype))
        final = final.to(device=self._cameraPoints.device)
        return final
