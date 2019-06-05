import torch
from threading import Thread
from .._ext import losses
from . import operations


class NormalLoss(torch.nn.Module):
    def __init__(self, nn_size=10):
        self.nn_size = nn_size

    def forward(self, pred, gt):
        pred_normals = operations.batch_normals(pred, nn_size=10, NCHW=True)
        gt_normals = operations.batch_normals(gt, nn_size=10, NCHW=True)
        # compare the normal with the closest point


class NmDistanceFunction(torch.autograd.Function):
    """3D point set to 3D point set distance"""
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        losses.nmdistance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


nndistance = NmDistanceFunction.apply


class ChamferLoss(torch.nn.Module):
    """
    chamfer loss. bidirectional nearest neighbor distance of two point sets.
    Args:
        threshold (float): distance beyond threshold*average_distance not be considered
        forward_weight (float): if != 1, different weight for chamfer distance forward and backward
        percentage (float): consider a percentage of inner points
    """

    def __init__(self, threshold=None, forward_weight=1.0, percentage=1.0):
        super(ChamferLoss, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.__threshold = threshold
        self.forward_weight = forward_weight
        self.percentage = percentage

    def set_threshold(self, value):
        self.__threshold = value

    def unset_threshold(self):
        self.__threshold = None

    def forward(self, pred, gt):
        """
        chamfer disntance between (B,N,3) and (B,M,3) points
        """
        assert(pred.dim() == 3 and gt.dim() == 3), \
            "input for ChamferLoss must be a 3D-tensor, but pred.size() is {} gt.size() is {}".format(pred.size(), gt.size())

        # need transpose
        if pred.size(2) != 3:
            assert(pred.size(1) == 3), "ChamferLoss is implemented for 3D points"
            pred = pred.transpose(2, 1).contiguous()

        if gt.size(2) != 3:
            assert(gt.size(1) == 3), "ChamferLoss is implemented for 3D points"
            gt = gt.transpose(2, 1).contiguous()

        assert(pred.size(2) == 3 and gt.size(2) ==
               3), "ChamferLoss is implemented for 3D points"

        if self.percentage < 1.0:
            # consider center points with higher weights
            pred_center = torch.mean(pred, dim=1, keepdim=True)
            num_point = pred.size(1)
            pred, _, _ = operations.group_knn(int(self.percentage * num_point), pred_center, pred, unique=False, NCHW=False)
            pred = torch.squeeze(pred, dim=1)
            # # BxN
            # dist_sqr = torch.sum((pred - pred_center)**2, dim=-1)
            # # Bx1
            # dist_sqrm = torch.max(dist_sqr, dim=1, keepdim=True)
            # weight = torch.exp(-dist_sqr / 1.5 * dist_sqrm)
            # weight = weight / torch.max(weight)
            # pred2gt = pred2gt * weight

            gt_center = torch.mean(gt, dim=1, keepdim=True)
            num_point = gt.size(1)
            gt, _, _ = operations.group_knn(int(self.percentage * num_point), gt_center, gt, unique=False, NCHW=False)
            gt = torch.squeeze(gt, dim=1)
            # # BxN
            # dist_sqr = torch.sum((label - label_center)**2, dim=-1)
            # # Bx1
            # dist_sqrm = torch.max(dist_sqr, dim=1, keepdim=True)
            # weight = torch.exp(-dist_sqr / 1.5 * dist_sqrm)
            # weight = weight / torch.max(weight)
            # gt2pred = gt2pred * weight

        pred2gt, gt2pred = nndistance(pred, gt)

        if self.__threshold is not None:
            threshold = self.__threshold
            forward_threshold = torch.mean(
                pred2gt, dim=1, keepdim=True) * threshold
            backward_threshold = torch.mean(
                gt2pred, dim=1, keepdim=True) * threshold
            # only care about distance within threshold (ignore strong outliers)
            pred2gt = torch.where(
                pred2gt < forward_threshold, pred2gt, torch.zeros_like(pred2gt))
            gt2pred = torch.where(
                gt2pred < backward_threshold, gt2pred, torch.zeros_like(gt2pred))

        # pred2gt is for each element in gt, the closest distance to this element
        pred2gt = torch.mean(pred2gt, dim=1)
        gt2pred = torch.mean(gt2pred, dim=1)
        CD_dist = self.forward_weight * pred2gt + gt2pred
        # CD_dist_norm = CD_dist/radius
        cd_loss = torch.mean(CD_dist)
        return cd_loss


if __name__ == '__main__':
    pc1 = torch.randn([2, 600, 3], dtype=torch.float64,
                      requires_grad=True).cuda()
    pc2 = torch.randn([2, 600, 3], dtype=torch.float64,
                      requires_grad=True).cuda()
    chamfer = ChamferLoss()
    from torch.autograd import gradcheck
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    pc2 = pc2.detach()
    test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    print(test)
