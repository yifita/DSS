# https://github.com/HTDerekLiu/Paparazzi/blob/master/utils/imageL0Smooth.py
# Referneces:
# 1. Xu et al. "Image Smoothing via L0 Gradient Minimization", 2011
# 2. This code is adapted from https://github.com/t-suzuki/l0_gradient_minimization_test

import numpy as np
from scipy.fftpack import fft2, ifft2
import skimage
from skimage.segmentation import slic
import torch


def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1: cols, ...] - imCum[:, 0: cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1: cols-r-1, ...]

    return imDst


def gf(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = scipy.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = scipy.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:, :, 0], r) / N
    mI_g = box(I[:, :, 1], r) / N
    mI_b = box(I[:, :, 2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:, :, 0]*p, r) / N
    mIp_g = box(I[:, :, 1]*p, r) / N
    mIp_b = box(I[:, :, 2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:, :, 0] * I[:, :, 0], r) / N - mI_r * mI_r
    var_I_rg = box(I[:, :, 0] * I[:, :, 1], r) / N - mI_r * mI_g
    var_I_rb = box(I[:, :, 0] * I[:, :, 2], r) / N - mI_r * mI_b

    var_I_gg = box(I[:, :, 1] * I[:, :, 1], r) / N - mI_g * mI_g
    var_I_gb = box(I[:, :, 1] * I[:, :, 2], r) / N - mI_g * mI_b

    var_I_bb = box(I[:, :, 2] * I[:, :, 2], r) / N - mI_b * mI_b

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]]
            ])
            covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
            a[i, j, :] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:, :, 0] * mI_r - a[:, :, 1] * mI_g - a[:, :, 2] * mI_b

    meanA = box(a, r) / N[..., np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = scipy.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = scipy.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def SuperPixel(images):
    # SLIC superpixel [Achanta et al. 2012]
    compactness = 20
    numSegments = 150
    maxIter = 3000
    imgSize = 256
    results = [None] * len(images)
    for idx, I in enumerate(images):
        isTensor = False
        if isinstance(I, torch.Tensor):
            device = I.device
            I = I.cpu().numpy()
            isTensor = True
        # compute FFT denominator (second part only)
        segs = skimage.segmentation.slic(I, compactness=compactness, n_segments=numSegments, enforce_connectivity=False)
        S = skimage.color.label2rgb(segs, I, kind='avg')
        if isTensor:
            results[idx] = torch.from_numpy(S).to(device=device, dtype=torch.float)
        else:
            results[idx] = S
    return results


def L0Smooth(images, lmd=0.05):
    results = []
    for idx, I in enumerate(images):
        betaMax = 1e5
        beta = 0.1
        betaRate = 2.0
        numIter = 40
        isTensor = False
        if isinstance(I, torch.Tensor):
            device = I.device
            I = I.cpu().numpy()
            isTensor = True
        # compute FFT denominator (second part only)
        FI = fft2(I, axes=(0, 1))
        dx = np.zeros((I.shape[0], I.shape[1]))  # gradient along x direction
        dy = np.zeros((I.shape[0], I.shape[1]))  # gradient along y direction
        dx[dx.shape[0]//2, dx.shape[1]//2-1:dx.shape[1]//2+1] = [-1, 1]
        dy[dy.shape[0]//2-1:dy.shape[0]//2+1, dy.shape[1]//2] = [-1, 1]
        denominator_second = np.conj(fft2(dx))*fft2(dx) + np.conj(fft2(dy))*fft2(dy)
        denominator_second = np.tile(np.expand_dims(denominator_second, axis=2), [1, 1, I.shape[2]])

        S = I
        hp = 0*I
        vp = 0*I
        for iter in range(numIter):
            # solve hp, vp
            hp = np.concatenate((S[:, 1:], S[:, :1]), axis=1) - S
            vp = np.concatenate((S[1:, :], S[:1, :]), axis=0) - S
            if len(I.shape) == 3:
                zeroIdx = np.sum(hp**2+vp**2, axis=2) < lmd/beta
            else:
                zeroIdx = hp**2.0 + vp**2.0 < lmd/beta
            hp[zeroIdx] = 0.0
            vp[zeroIdx] = 0.0

            # solve S
            hv = np.concatenate((hp[:, -1:], hp[:, :-1]), axis=1) - hp + np.concatenate((vp[-1:, :], vp[:-1, :]), axis=0) - vp
            S = np.real(ifft2((FI + (beta*fft2(hv, axes=(0, 1)))) / (1+beta*denominator_second), axes=(0, 1)))

            # update parameters
            beta *= betaRate
            if beta > betaMax:
                break
        if isTensor:
            results.append(torch.from_numpy(S).to(device=device, dtype=torch.float))
        else:
            results.append(S)
    return results


def Pix2PixDenoising(images, model=None):
    import os
    from .pix2pix.options.test_options import TestOptions
    from .pix2pix.data import create_dataset
    from .pix2pix.models import create_model
    import torch
    opt = TestOptions().parse()  # get test options
    opt.gpu_ids = [torch.cuda.current_device()]
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = 'trained_models'
    # opt.name = 'render_PCA_resnet'
    opt.name = model or 'render_PCA_resnet_noise03_knn15'
    opt.epoch = "latest"
    # opt.name = 'render_test_25shape_reconv_pix'
    opt.norm = 'pixel'
    # opt.netG = 'unet_256_Re1'
    opt.netG = 'resnet_9blocks'

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    images = torch.stack(images, dim=0)
    with torch.no_grad():
        # B, 3, W, H
        images = images.permute(0, 3, 1, 2)
        images_normalized = images - 0.5
        input_dict = {'A': images_normalized, 'A_paths': [None]}
        model.set_input(input_dict)  # unpack data from data loader
        model.test()                 # run inference
        results = model.get_current_visuals()  # get image results
        B, C, H, W = results['real_A'].shape
        minValues = torch.min(results['real_A'].view(B, C, -1), dim=2, keepdim=True)[0].view(B, C, 1, 1)
        maxValues = torch.max(results['real_A'].view(B, C, -1), dim=2, keepdim=True)[0].view(B, C, 1, 1)
        results['fake_B'] = torch.min(results['fake_B'], maxValues.expand(-1, -1, H, W))
        # results['real_A'] = torch.min(results['real_A'], maxValues.expand(-1, -1, H, W))
        results['fake_B'] = torch.max(results['fake_B'], minValues.expand(-1, -1, H, W))
        # results['real_A'] = torch.max(results['real_A'], minValues.expand(-1, -1, H, W))

        results = results["fake_B"] + 0.5
        results = results.permute(0, 2, 3, 1).contiguous()
    return results
