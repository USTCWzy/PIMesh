import torch
import torch.nn as nn

import numpy as np

# from pytorch3d.structures import Meshes
from utils.geometry.geometry import batch_rodrigues
from utils.joints.evaluate import joint_mapping

class HPSPILoss(nn.Module):
    def __init__(self,
                 camera,
                 camera_silh,
                 renderer_silhouette,
                 faces,
                 joint_mode='limb_shoulder_nose',
                 bed_depth=1.66,
                 sensor_pitch=[0.0195, 0.0311],
                 e_pressure_weight=1.,
                 e_3d_loss_weight=30.,
                 e_2d_loss_weight=60.,
                 e_pose_loss_weight=1.,
                 e_shape_loss_weight=0.01,
                 e_trans_loss_weight=100.,
                 e_smooth_pose_weight=10.,
                 e_smooth_betas_weight=10.,
                 e_smooth_trans_weight=10.,
                 e_smooth_joints_weight=10.,
                 e_verts_pen_weight=100.,
                 device='cuda'
                 ):
        super(HPSPILoss, self).__init__()
        self.e_pressure_weight = e_pressure_weight

        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_2d_loss_weight = e_2d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.e_trans_loss_weight = e_trans_loss_weight

        self.e_smooth_pose_weight = e_smooth_pose_weight
        self.e_smooth_betas_weight = e_smooth_betas_weight
        self.e_smooth_trans_weight = e_smooth_trans_weight
        self.e_smooth_joints_weight = e_smooth_joints_weight

        self.e_verts_pen_weight = e_verts_pen_weight

        self.device = device
        self.camera = camera.to(device)
        # self.camera_silh = camera_silh.to(device)
        # self.render_silhoule = renderer_silhouette.to(device)
        self.faces = faces.to(device)

        self.bed_depth = torch.tensor([0, 0, bed_depth]).to(device)
        self.sensor_pitch = sensor_pitch.to(device)

        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        self.smpl_joints_index, self.label_index = joint_mapping('limb_shoulder_nose')

    def forward(self,
                outputs,
                gt_keypoints_3d,
                gt_keypoints_2d,
                gt_betas,
                gt_pose,
                gt_trans,
                gt_pressure_binary
                ):
        # to reduce time dimension
        # import pdb;pdb.set_trace()
        loss_smooth_joint = torch.tensor(0.0).to(self.device)
        loss_smooth_pose = torch.tensor(0.0).to(self.device)
        loss_smooth_betas = torch.tensor(0.0).to(self.device)
        loss_smooth_trans = torch.tensor(0.0).to(self.device)
        loss_press_proj = torch.tensor(0.0).to(self.device)
        loss_kp_2d = torch.tensor(0.0).to(self.device)

        if gt_trans.shape[1] > 1:

            loss_smooth_joint = self.batch_smooth_joint_loss(outputs['kp_3d'])
            loss_smooth_pose = self.batch_smooth_pose_loss(outputs['theta'][:, :, 3:75])
            loss_smooth_betas = self.batch_smooth_shape_loss(outputs['theta'][:, :, 75:])
            loss_smooth_trans = self.batch_smooth_trans_loss(outputs['theta'][:, :, :3])

        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

        gt_keypoints_3d = reduce(gt_keypoints_3d)
        # gt_keypoints_2d = reduce(gt_keypoints_2d)
        gt_betas = reduce(gt_betas)
        gt_pose = reduce(gt_pose)
        gt_trans = reduce(gt_trans)
        # gt_pressure_binary = reduce(gt_pressure_binary)

        for key in outputs:
            outputs[key] = reduce(outputs[key])

        # import pdb;pdb.set_trace()

        # loss_kp_2d = self.keypoints_2d(outputs['kp_3d'][:, self.smpl_joints_index, :2], gt_keypoints_2d[:, self.label_index])
        loss_kp_3d = self.keypoints_3d(outputs['kp_3d'][:, :, :3], gt_keypoints_3d)

        # loss_press_proj = self.pressure_projection(outputs['verts'], gt_pressure_binary)


        loss_smpl_pose = self.loss_smpl_pose(outputs['theta'][:, 3:75], gt_pose)
        loss_smpl_shape = self.loss_smpl_shape(outputs['theta'][:, 75:], gt_betas)
        loss_smpl_trans = self.loss_smpl_trans(outputs['theta'][:, :3], gt_trans)

        # loss_verts_pen = self.pen_loss(outputs['verts'])
        loss_verts_pen = torch.tensor(0.0).to(self.device)

        # loss = self.e_2d_loss_weight * loss_kp_2d + \
        #        self.e_3d_loss_weight * loss_kp_3d + \
        #        self.e_pose_loss_weight * loss_smpl_pose + \
        #        self.e_shape_loss_weight * loss_smpl_shape + \
        #        self.e_trans_loss_weight * loss_smpl_trans + \
        #        self.e_smooth_joints_weight * loss_smooth_joint + \
        #        self.e_smooth_pose_weight * loss_smooth_pose + \
        #        self.e_smooth_betas_weight * loss_smooth_betas + \
        #        self.e_smooth_trans_weight * loss_smooth_trans + \
        #        self.e_pressure_weight * loss_press_proj + \
        #        self.e_verts_pen_weight * loss_verts_pen

        loss = self.e_pose_loss_weight * loss_smpl_pose + \
               self.e_shape_loss_weight * loss_smpl_shape + \
               self.e_trans_loss_weight * loss_smpl_trans + \
               self.e_3d_loss_weight * loss_kp_3d + \
               self.e_smooth_joints_weight * loss_smooth_joint + \
               self.e_smooth_pose_weight * loss_smooth_pose + \
               self.e_smooth_betas_weight * loss_smooth_betas + \
               self.e_smooth_trans_weight * loss_smooth_trans

        mpjpe_loss = self.mpjpe(outputs['kp_3d'][:, :, :3], gt_keypoints_3d)
        # loss = 1000 * mpjpe_loss

        
        loss_dict = {
            't_mpjpe': mpjpe_loss.item(),
            'kp2d': loss_kp_2d.item(),
            'kp3d': loss_kp_3d.item(),
            'trans': loss_smpl_trans.item(),
            'pose': loss_smpl_pose.item(),
            'shape': loss_smpl_shape.item(),
            'sm_kp': loss_smooth_joint.item(),
            'sm_ps': loss_smooth_pose.item(),
            'sm_bt': loss_smooth_betas.item(),
            'sm_ts': loss_smooth_trans.item(),
            'proj': loss_press_proj.item(),
            'pen': loss_verts_pen.item()
        }

        # if loss.item() > 1e7:
        #     import pdb;pdb.set_trace()
        #
        # if loss_dict['kp2d'] > 100 and loss_dict['t_mpjpe'] < 0.1:
        #     import pdb;
        #     pdb.set_trace()
        
        return loss, loss_dict


    def keypoints_2d(self, pred_keypoints_2d, gt_keypoints_2d):
        pred_keypoints_pi_coord = pred_keypoints_2d * self.sensor_pitch
        return self.criterion_keypoints(pred_keypoints_pi_coord, gt_keypoints_2d).mean()

    def keypoints_3d(self, pred_keypoints_3d, gt_keypoints_3d):
        # gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        # pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
        # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()

    # def pressure_projection(self, verts, gt_pressure_binary):
    #     verts *= torch.tensor([1, -1, 1]).to(self.device)
    #     verts[:, :, 2] = verts[:, :, 2] + torch.tensor(
    #         0.03
    #     ).to(self.device)
    #
    #     meshes_world = Meshes(verts=verts[:, :, :], faces=self.faces[:verts.shape[0]])
    #     images_predicted = self.render_silhoule(meshes_world, cameras=self.camera_silh, lights=None)
    #
    #     return torch.abs(images_predicted[:, :, :, -1] - gt_pressure_binary).mean()

    def loss_smpl_pose(self, pred_pose, gt_pose):
        pred_rotmat_valid = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        return self.criterion_shape(pred_rotmat_valid, gt_rotmat_valid)

    def loss_smpl_shape(self, pred_shape, gt_shape):
        return self.criterion_shape(pred_shape, gt_shape)

    def loss_smpl_trans(self, pred_trans, gt_trans):
        return self.criterion_shape(pred_trans, gt_trans)

    def batch_smooth_pose_loss(self, pred_pose):
        pose_diff = pred_pose[:, 1:] - pred_pose[:, :-1]
        return torch.mean(pose_diff.abs())

    def batch_smooth_joint_loss(self, pred_joint):
        pose_diff = pred_joint[:, 1:] - pred_joint[:, :-1]
        return torch.mean(pose_diff.abs())

    def batch_smooth_shape_loss(self, pred_betas):
        shape_diff = pred_betas[:, 1:] - pred_betas[:, :-1]
        return torch.mean(shape_diff.abs())

    def batch_smooth_trans_loss(self, pred_trans):
        trans_diff = pred_trans[:, 1:] - pred_trans[:, :-1]
        return torch.mean(trans_diff.abs())

    def pen_loss(self, verts):
        inside_verts = verts[:, :, 2] > 0
        outside_verts = (verts[:, :, 2] > - 0.03) & (~inside_verts)

        v2boutside = 1 * torch.tanh(torch.abs((verts[:, :, 2][outside_verts])) / 0.15) ** 2
        v2binside = 0.005 * torch.tanh((verts[:, :, 2][inside_verts]) / 0.002) ** 2

        bed_contact_ver_loss = v2binside.mean() + v2boutside.mean()

        return bed_contact_ver_loss

    def mpjpe(self, pred_keypoints_3d, gt_keypoints_3d):

        return ((pred_keypoints_3d - gt_keypoints_3d)**2).sum(-1).sqrt().mean()
