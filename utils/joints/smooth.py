import time
import math
import torch
import warnings
import numpy as np
from scipy import signal
from .one_euro import *

def create_smooth(args):
    if args is None:
        return joint_without_smooth
    elif args['smooth_type'] == 'one_euro':
        return joint_one_euro_filter
    elif args['smooth_type'] == 'savgol':
        return joint_savgol_filter
    elif args['smooth_type'] == 'gaussian1d':
        return joint_gaussian_1d_filter
def joint_without_smooth(args, key_points):
    return key_points
def joint_savgol_filter(args, key_points):
    '''
    2d joint savgol filter
    input:
        args(dict):
            windows,
            polyorder,
        key_points(np.array): No. frames * keypoint_num * 3
    '''
    (N, K, D) = key_points.shape
    smooth = np.zeros(key_points.shape)
    key_point_type = key_points
    if isinstance(key_points, torch.Tensor):
        if key_points.is_cuda:
            key_points = key_points.cpu().numpy()
        else:
            key_points = key_points.numpy()
    # smooth at different axis
    start = time.time()
    for i in range(D):
        smooth[..., i] = signal.savgol_filter(
            key_points[..., i], args['window_length'], args['polyorder'], axis=0)
    inference_time = time.time() - start
    print('smooth time:', inference_time)
    if isinstance(key_point_type, torch.Tensor):
        # we also return tensor by default
        if key_point_type.is_cuda:
            smooth = torch.from_numpy(smooth).cuda()
        else:
            smooth = torch.from_numpy(smooth)
    return smooth

def joint_gaussian_1d_filter(args, key_points):
    '''
    2d joint gaussian 1d filter
    input:
        args(dict):
            windows,
            sigma,
        key_points(np.array): No. frames * keypoint_num * 3
    '''
    (N, K, D) = key_points.shape
    smooth = np.zeros(key_points.shape)
    key_point_type = key_points
    if isinstance(key_points, torch.Tensor):
        if key_points.is_cuda:
            key_points = key_points.cpu().numpy()
        else:
            key_points = key_points.numpy()
    # smooth at different axis
    start = time.time()
    for i in range(D):
        smooth[..., i] = signal.savgol_filter(
            key_points[..., i], args['window_length'], args['polyorder'], axis=0)
    inference_time = time.time() - start
    print('smooth time:', inference_time)
    if isinstance(key_point_type, torch.Tensor):
        # we also return tensor by default
        if key_point_type.is_cuda:
            smooth = torch.from_numpy(smooth).cuda()
        else:
            smooth = torch.from_numpy(smooth)
    return smooth

def joint_one_euro_filter(args, key_points):
    # x (np.ndarray): input poses.
    if len(key_points.shape) != 3:
        warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
    assert len(key_points.shape) == 3
    x_type = key_points
    if isinstance(key_points, torch.Tensor):
        if key_points.is_cuda:
            key_points = key_points.cpu().numpy()
        else:
            key_points = key_points.numpy()

    one_euro_filter = OneEuro(
        np.zeros_like(key_points[0]),
        key_points[0],
        min_cutoff=args['min_cutoff'],
        beta=args['beta'],
    )

    pred_pose_hat = np.zeros_like(key_points)

    # initialize
    pred_pose_hat[0] = key_points[0]

    start = time.time()
    for idx, pose in enumerate(key_points[1:]):
        idx += 1
        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose
    end = time.time()
    inference_time = end - start

    if isinstance(x_type, torch.Tensor):
        # we also return tensor by default
        if x_type.is_cuda:
            pred_pose_hat = torch.from_numpy(pred_pose_hat).cuda()
        else:
            pred_pose_hat = torch.from_numpy(pred_pose_hat)
    return pred_pose_hat
