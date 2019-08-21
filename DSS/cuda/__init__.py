import torch
from .rasterize_forward import _guided_scatter_maps, _scatter_maps, _gather_maps, _compute_visibility_maps
from .rasterize_backward import _visibility_backward, _visibility_reference_backward

guided_scatter_maps = _guided_scatter_maps
scatter_maps =  _scatter_maps

__all__ = ["rasterizeDSS", "rasterizeRBF", "gather_maps", "guided_scatter_maps", "scatter_maps"]


class GatherMaps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indiceMap, defaultValue):
        ctx.numPoint = values.shape[1]
        ctx.save_for_backward(indiceMap)
        gathered = _gather_maps(values, indiceMap, defaultValue)
        return gathered

    @staticmethod
    def backward(ctx, grad):
        indiceMap, = ctx.saved_tensors
        dIn = _scatter_maps(ctx.numPoint, grad, indiceMap)
        return dIn, None, None

gather_maps = GatherMaps.apply

class RasterizeRBFBaselineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, rhoValues, Ws, projPoints, boundingBoxes, inplane, Ms, cameraPoints,
                width, height, localWidth, localHeight, camFar, focalLength, mergeThreshold, considerZ, topK):
        """
        input:
            rho            BxNxhxw   rho evaluated in a BB of (h, w) around projected points
            rhoValues      BxNx1     normalizing term of rho
            Ws             BxNx3     point color
            projPoints     BxNx2or3  projected point location
            boundingBoxes  BxNx4     bounding boxes (w0, h0, width, height)
            inplane        BxNxhxwx3 window back projected in space
            Ms             BxNx2x2   inverse of gaussian variance
            cameraPoints   BxNx3     point location in camera space
            width          scalar
            height         scalar
            localWidth     scalar    used to limit gradient computation to local window
            localHeight    scalar
            carFar         scalar
            focalLength    scalar
            mergeThreshold scalar    depth merging threshold T
            considerZ      bool      consider Z gradient
            topK           int       front most points to consider during forward/backward rendering
        returns:
            pixels         BxHxWx3
            pointIdxMap    BxHxWx5
            rhoMap         BxHxWx5
            WsMap          BxHxWx5x3
            isBehind       BxHxWx5
        """
        batchSize, numPoints, bbHeight, bbWidth = rho.shape
        # compute visiblity, return per pixel top5 contributor sorted by their
        # depthValue
        # pointIdxMap BxHxWx5, index inside the splat's bounding box window which is to be rendered at pixel (h,w)
        # depthMap    BxHxWx5, depths of splats that are rendered at pixel (h,w)
        # rhoMap      BxHxWx5
        pointIdxMap = torch.full((batchSize, height, width, topK), -1, dtype=torch.int64, device=rho.device)
        depthMap = torch.full((batchSize, height, width, topK), camFar, dtype=rho.dtype, device=rho.device)
        bbPositionMap = torch.full((batchSize, height, width, topK, 2), -1, dtype=torch.int64, device=rho.device)
        with torch.cuda.device(rho.device):
            # call visibility kernel, outputs depthMap, pointIdxMap which store the depth and index of
            # the 5 closest point for each pixel, if less than 5 points paint the pixel, set idxMap to -1
            _compute_visibility_maps(boundingBoxes[:, :, :2].contiguous(), inplane, pointIdxMap, bbPositionMap, depthMap)
            # gather rho, wk
            WsMap = _gather_maps(Ws, pointIdxMap, 0.0)
            # per batch indice for rhos bx(Nxhxw)
            # gather rho, Ws values from pointIdxMap and bbPositionMap, if idx < 0 (unset), then set rho=0 Ws=0
            totalIdxMap = pointIdxMap*bbHeight*bbWidth+bbPositionMap[:, :, :, :, 0]*bbWidth+bbPositionMap[:, :, :, :, 1]
            validMaps = totalIdxMap >= 0
            totalIdxMap = torch.where(validMaps, totalIdxMap, torch.full_like(totalIdxMap, -1))
            rhoMap = _gather_maps(rho.reshape(batchSize, -1, 1), totalIdxMap, 0.0).squeeze(-1)
            # check depth jump
            isBehind = torch.zeros(depthMap.shape, dtype=torch.uint8, device=depthMap.device)
            isBehind[:, :, :, 1:] = (depthMap[:, :, :, 1:]-depthMap[:, :, :, :1]) > mergeThreshold
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            # WsMap[:, :, :, 1:, :] = torch.where(isBehind.unsqueeze(-1), torch.zeros(1, 1, 1, 1, 1, device=WsMap.device, dtype=WsMap.dtype), WsMap[:, :, :, 1:])
            # normalize rho
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            sumRho = torch.where(sumRho == 0, torch.ones_like(sumRho), sumRho)
            rhoMap_normalized = rhoMap_filtered/sumRho
            # rho * w
            pixels = torch.sum(WsMap * rhoMap_normalized.unsqueeze(-1), dim=3)
            # accumulated = WsMap[:, :, :, 0, :]
            ctx.save_for_backward(pointIdxMap, bbPositionMap, isBehind, WsMap, rhoMap, depthMap, Ws, rhoValues, projPoints, cameraPoints, boundingBoxes, pixels, Ms)
            ctx.numPoint = numPoints
            ctx.bbWidth = bbWidth
            ctx.bbHeight = bbHeight
            ctx.localHeight = localHeight
            ctx.localWidth = localWidth
            ctx.mergeThreshold = mergeThreshold
            ctx.focalLength = focalLength
            ctx.considerZ = considerZ
            # ctx.repulsion_radius = repulsion_radius
            # ctx.repulsion_weight = repulsion_weight
            ctx.rho_requires_grad = rho.requires_grad
            ctx.w_requires_grad = Ws.requires_grad
            ctx.xyz_requires_grad = projPoints.requires_grad
            ctx.mark_non_differentiable(pointIdxMap, rhoMap, WsMap, isBehind)
            return pixels, pointIdxMap, rhoMap_normalized, WsMap, isBehind

    @staticmethod
    def backward(ctx, gradPixels, dpointIdxMap, gradRhoMap, gradWsMap, gradIsBehind):
        """
        input
            gradPixels          (BxHxWx3)
        output
            dRho            (BxNxbbHxbbW)
            dW              (BxNx3)
            dP              (BxNx2) derivative wrt projected points
            dcamP           (BxNx3) derivative wrt camera points (only z-dim is nonzero)
        """
        pointIdxMap, bbPositionMap, isBehind, WsMap, rhoMap, depthMap, Ws, rhoValues, projPoints, cameraPoints, boundingBoxes, pixels, Ms = ctx.saved_tensors
        mergeThreshold = ctx.mergeThreshold
        focalLength = ctx.focalLength
        numPoint = ctx.numPoint
        considerZ = ctx.considerZ
        bbWidth = ctx.bbWidth
        bbHeight = ctx.bbHeight
        batchSize, height, width, topK, C = WsMap.shape
        if ctx.needs_input_grad[0]:  # rho will not be backpropagated
            WsMap_ = torch.where(isBehind.unsqueeze(-1), torch.zeros(1, 1, 1, 1, 1, device=WsMap.device, dtype=WsMap.dtype), WsMap)
            totalIdxMap = pointIdxMap*bbHeight*bbWidth+bbPositionMap[:, :, :, :, 0]*bbWidth+bbPositionMap[:, :, :, :, 1]
            # TODO check dNormalizeddRho
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            dNormalizeddRho = torch.where(rhoMap > 0, 1/sumRho-rhoMap/sumRho, rhoMap)
            dRho = _guided_scatter_maps(numPoint*bbWidth*bbHeight, dNormalizeddRho.unsqueeze(-1)*gradPixels.unsqueeze(3)*WsMap_, totalIdxMap, boundingBoxes)
            dRho = torch.sum(dRho, dim=-1)
            dRho = torch.reshape(dRho, (batchSize, numPoint, bbHeight, bbWidth))
        else:
            dRho = None

        if ctx.needs_input_grad[2]:
            # dPixels/dWs = Rho
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            sumRho = torch.where(sumRho == 0, torch.zeros_like(sumRho), sumRho)
            rhoMap_normalized = rhoMap_filtered/sumRho
            # BxHxWx3 -> BxHxWxKx3 -> BxNx3
            dWs = _guided_scatter_maps(numPoint, gradPixels.unsqueeze(3)*rhoMap_normalized.unsqueeze(-1), pointIdxMap, boundingBoxes)
        else:
            dWs = None

        if ctx.needs_input_grad[3]:
            localWidth = ctx.localWidth
            localHeight = ctx.localHeight
            depthValues = cameraPoints[:, :, 2].contiguous()
            # B,N,1
            dIdp = torch.zeros_like(projPoints, device=gradPixels.device, dtype=gradPixels.dtype)
            dIdz = torch.zeros(1, numPoint, device=gradPixels.device, dtype=gradPixels.dtype)
            gamma = 0.1
            # rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            # sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            # rhoMap_normalized = rhoMap_filtered/sumRho
            # rhoMap_normalized = rhoMap_filtered/sumRho
            outputs = _visibility_reference_backward(focalLength, mergeThreshold, gamma, considerZ, localHeight, localWidth, 0,
                                                            gradPixels, pointIdxMap, rhoMap, WsMap, depthMap, isBehind,
                                                            pixels, boundingBoxes, projPoints, Ws, depthValues, rhoValues, Ms, dIdp, dIdz)
            dIdp, dIdz, debugTensor = outputs
            dIdcam = torch.zeros_like(cameraPoints)
            dIdcam[:, :, 2] = dIdz
            # saved_variables["dI"] = gradPixels.detach().cpu()
            # saved_variables["dIdp"] = saved_variables["dIdp"].scatter_(1, saved_variables["renderable_idx"].expand(-1, -1, dIdp.shape[-1]),
            #                                                            dIdp.cpu().detach())
            # saved_variables["projPoints"] = saved_variables["projPoints"].scatter_(1, saved_variables["renderable_idx"].expand(-1,-1,dIdp.shape[-1]),
            #                                                                        projPoints.cpu().detach())
            # saved_variables["dIdpMap"] = debugTensor[:,:,:,:2].cpu().detach()
        else:
            dIdp = dIdcam = None

        return (None, None, dWs, dIdp, None, None, dIdcam, None, None, None, None, None, None, None, None, None, None)
class RasterizeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, rhoValues, Ws, projPoints, boundingBoxes, inplane, Ms, cameraPoints,
                width, height, localWidth, localHeight, camFar, focalLength, mergeThreshold, considerZ, topK):
        """
        input:
            rho            BxNxhxw   rho evaluated in a BB of (h, w) around projected points
            rhoValues      BxNx1     normalizing term of rho
            Ws             BxNx3     point color
            projPoints     BxNx2or3  projected point location
            boundingBoxes  BxNx4     bounding boxes (w0, h0, width, height)
            inplane        BxNxhxwx3 window back projected in space
            Ms             BxNx2x2   inverse of gaussian variance
            cameraPoints   BxNx3     point location in camera space
            width          scalar
            height         scalar
            localWidth     scalar    used to limit gradient computation to local window
            localHeight    scalar
            carFar         scalar
            focalLength    scalar
            mergeThreshold scalar    depth merging threshold T
            considerZ      bool      consider Z gradient
            topK           int       front most points to consider during forward/backward rendering
        returns:
            pixels         BxHxWx3
            pointIdxMap    BxHxWx5
            rhoMap         BxHxWx5
            WsMap          BxHxWx5x3
            isBehind       BxHxWx5
        """
        batchSize, numPoints, bbHeight, bbWidth = rho.shape
        # compute visiblity, return per pixel top5 contributor sorted by their
        # depthValue
        # pointIdxMap BxHxWx5, index inside the splat's bounding box window which is to be rendered at pixel (h,w)
        # depthMap    BxHxWx5, depths of splats that are rendered at pixel (h,w)
        # rhoMap      BxHxWx5
        pointIdxMap = torch.full((batchSize, height, width, topK), -1, dtype=torch.int64, device=rho.device)
        depthMap = torch.full((batchSize, height, width, topK), camFar, dtype=rho.dtype, device=rho.device)
        bbPositionMap = torch.full((batchSize, height, width, topK, 2), -1, dtype=torch.int64, device=rho.device)
        with torch.cuda.device(rho.device):
            # call visibility kernel, outputs depthMap, pointIdxMap which store the depth and index of
            # the 5 closest point for each pixel, if less than 5 points paint the pixel, set idxMap to -1
            _compute_visibility_maps(boundingBoxes[:, :, :2].contiguous(), inplane, pointIdxMap, bbPositionMap, depthMap)
            # gather rho, wk
            WsMap = _gather_maps(Ws, pointIdxMap, 0.0)
            # per batch indice for rhos bx(Nxhxw)
            # gather rho, Ws values from pointIdxMap and bbPositionMap, if idx < 0 (unset), then set rho=0 Ws=0
            totalIdxMap = pointIdxMap*bbHeight*bbWidth+bbPositionMap[:, :, :, :, 0]*bbWidth+bbPositionMap[:, :, :, :, 1]
            validMaps = totalIdxMap >= 0
            totalIdxMap = torch.where(validMaps, totalIdxMap, torch.full_like(totalIdxMap, -1))
            rhoMap = _gather_maps(rho.reshape(batchSize, -1, 1), totalIdxMap, 0.0).squeeze(-1)
            # check depth jump
            isBehind = torch.zeros(depthMap.shape, dtype=torch.uint8, device=depthMap.device)
            isBehind[:, :, :, 1:] = (depthMap[:, :, :, 1:]-depthMap[:, :, :, :1]) > mergeThreshold
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            # WsMap[:, :, :, 1:, :] = torch.where(isBehind.unsqueeze(-1), torch.zeros(1, 1, 1, 1, 1, device=WsMap.device, dtype=WsMap.dtype), WsMap[:, :, :, 1:])
            # normalize rho
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            sumRho = torch.where(sumRho == 0, torch.ones_like(sumRho), sumRho)
            rhoMap_normalized = rhoMap_filtered/sumRho
            # rho * w
            pixels = torch.sum(WsMap * rhoMap_normalized.unsqueeze(-1), dim=3)
            # accumulated = WsMap[:, :, :, 0, :]
            ctx.save_for_backward(pointIdxMap, bbPositionMap, isBehind, WsMap, rhoMap, depthMap, Ws, rhoValues, projPoints, cameraPoints, boundingBoxes, pixels, Ms)
            ctx.numPoint = numPoints
            ctx.bbWidth = bbWidth
            ctx.bbHeight = bbHeight
            ctx.localHeight = localHeight
            ctx.localWidth = localWidth
            ctx.mergeThreshold = mergeThreshold
            ctx.focalLength = focalLength
            ctx.considerZ = considerZ
            ctx.rho_requires_grad = rho.requires_grad
            ctx.w_requires_grad = Ws.requires_grad
            ctx.xyz_requires_grad = projPoints.requires_grad
            ctx.mark_non_differentiable(pointIdxMap, rhoMap, WsMap, isBehind)
            return pixels, pointIdxMap, rhoMap_normalized, WsMap, isBehind

    @staticmethod
    def backward(ctx, gradPixels, dpointIdxMap, gradRhoMap, gradWsMap, gradIsBehind):
        """
        input
            gradPixels          (BxHxWx3)
        output
            dRho            (BxNxbbHxbbW)
            dW              (BxNx3)
            dP              (BxNx2) derivative wrt projected points
            dcamP           (BxNx3) derivative wrt camera points (only z-dim is nonzero)
        """
        pointIdxMap, bbPositionMap, isBehind, WsMap, rhoMap, depthMap, Ws, rhoValues, projPoints, cameraPoints, boundingBoxes, pixels, Ms = ctx.saved_tensors
        mergeThreshold = ctx.mergeThreshold
        focalLength = ctx.focalLength
        numPoint = ctx.numPoint
        considerZ = ctx.considerZ
        bbWidth = ctx.bbWidth
        bbHeight = ctx.bbHeight
        batchSize, height, width, topK, C = WsMap.shape
        if ctx.needs_input_grad[0]:  # rho will not be backpropagated
            WsMap_ = torch.where(isBehind.unsqueeze(-1), torch.zeros(1, 1, 1, 1, 1, device=WsMap.device, dtype=WsMap.dtype), WsMap)
            totalIdxMap = pointIdxMap*bbHeight*bbWidth+bbPositionMap[:, :, :, :, 0]*bbWidth+bbPositionMap[:, :, :, :, 1]
            # TODO check dNormalizeddRho
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            dNormalizeddRho = torch.where(rhoMap > 0, 1/sumRho-rhoMap/sumRho, rhoMap)
            dRho = _guided_scatter_maps(numPoint*bbWidth*bbHeight, dNormalizeddRho.unsqueeze(-1)*gradPixels.unsqueeze(3)*WsMap_, totalIdxMap, boundingBoxes)
            dRho = torch.sum(dRho, dim=-1)
            dRho = torch.reshape(dRho, (batchSize, numPoint, bbHeight, bbWidth))
        else:
            dRho = None

        if ctx.needs_input_grad[2]:
            # dPixels/dWs = Rho
            rhoMap_filtered = torch.where(isBehind, torch.zeros(1, 1, 1, 1, device=rhoMap.device, dtype=rhoMap.dtype), rhoMap)
            sumRho = torch.sum(rhoMap_filtered, dim=-1, keepdim=True)
            sumRho = torch.where(sumRho == 0, torch.zeros_like(sumRho), sumRho)
            rhoMap_normalized = rhoMap_filtered/sumRho
            # BxHxWx3 -> BxHxWxKx3 -> BxNx3
            dWs = _guided_scatter_maps(numPoint, gradPixels.unsqueeze(3)*rhoMap_normalized.unsqueeze(-1), pointIdxMap, boundingBoxes)
        else:
            dWs = None

        if ctx.needs_input_grad[3]:
            localWidth = ctx.localWidth
            localHeight = ctx.localHeight
            depthValues = cameraPoints[:, :, 2].contiguous()
            # B,N,1
            dIdp = torch.zeros_like(projPoints, device=gradPixels.device, dtype=gradPixels.dtype)
            dIdz = torch.zeros(1, numPoint, device=gradPixels.device, dtype=gradPixels.dtype)
            outputs = _visibility_backward(focalLength, mergeThreshold, considerZ,
                                                             localHeight, localWidth,
                                                             gradPixels, pointIdxMap, rhoMap, WsMap, depthMap, isBehind,
                                                             pixels, boundingBoxes, projPoints, Ws, depthValues, rhoValues, dIdp, dIdz)
            # outputs = rasterize_backward.visibility_debug_backward(mergeThreshold, focalLength, considerZ,
            #                                                        localHeight, localWidth, 0,
            #                                                        gradPixels, pointIdxMap, rhoMap, WsMap, depthMap, isBehind,
            #                                                        pixels, boundingBoxes, projPoints, Ws, depthValues, rhoValues, dIdp, dIdz)
            # dIdp, dIdz = outputs
            dIdcam = torch.zeros_like(cameraPoints)
            dIdcam[:, :, 2] = dIdz
        else:
            dIdp = dIdcam = None

        return (None, None, dWs, dIdp, None, None, dIdcam, None, None, None, None, None, None, None, None, None, None)


def rasterizeDSS(rho, rhoValues, Ws, projPoints, boundingBoxes, inplane, Ms, cameraPoints,
              width, height, camFar, focalLength, localWidth=None, localHeight=None,
              mergeThreshold=0.05, considerZ=False, topK=5):
    d = torch.cuda.current_device()
    localWidth = localWidth or 2*width
    localHeight = localHeight or 2*height
    return RasterizeAutograd.apply(rho.to(device=d), rhoValues.to(device=d), Ws.to(device=d), projPoints.to(device=d), boundingBoxes.to(device=d),
                                   inplane.to(device=d), Ms.to(device=d), cameraPoints.to(device=d),
                                   width, height, localWidth, localHeight, camFar, focalLength, mergeThreshold, considerZ, topK)



def rasterizeRBF(rho, rhoValues, Ws, projPoints, boundingBoxes, inplane, Ms, cameraPoints,
              width, height, camFar, focalLength, localWidth=None, localHeight=None,
              mergeThreshold=0.05, considerZ=False, topK=5):
    d = torch.cuda.current_device()
    localWidth = localWidth or 2*width
    localHeight = localHeight or 2*height
    return RasterizeRBFBaselineAutograd.apply(rho.to(device=d), rhoValues.to(device=d), Ws.to(device=d), projPoints.to(device=d), boundingBoxes.to(device=d),
                                   inplane.to(device=d), Ms.to(device=d), cameraPoints.to(device=d),
                                   width, height, localWidth, localHeight, camFar, focalLength, mergeThreshold, considerZ, topK)


