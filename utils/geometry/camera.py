# copied from smplify-xmc

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=5000, focal_length_y=5000,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        # create focal length parameter
        focal_length_x = focal_length_x * torch.ones([batch_size, 1], dtype=dtype)
        focal_length_x = nn.Parameter(focal_length_x, requires_grad=True)
        self.register_parameter('focal_length_x', focal_length_x)
        focal_length_y = focal_length_y * torch.ones([batch_size, 1], dtype=dtype)
        focal_length_y = nn.Parameter(focal_length_y, requires_grad=True)
        self.register_parameter('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)

        pitch = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('pitch', pitch)
        roll = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('roll', roll)
        yaw = nn.Parameter(torch.zeros([batch_size,1], dtype=dtype),
            requires_grad=True)
        self.register_parameter('yaw', yaw)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        translation = nn.Parameter(translation,
            requires_grad=True)
        self.register_parameter('translation', translation)

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def ea2rm(self):
        x = self.pitch
        y = self.yaw
        z = self.roll
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)

        R = torch.stack(
              [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
               torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
               torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)

        return R

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
        camera_mat[:, 0, 0] = self.focal_length_x.flatten()
        camera_mat[:, 1, 1] = self.focal_length_y.flatten()

        rotation = self.ea2rm()
        self.rotation[:] = rotation.detach()
        camera_transform = transform_mat(rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center

        return img_points
def perspective_projection(points, rotation, translation,
                           focal_length_x, focal_length_y, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length_x
    K[:,1,1] = focal_length_y
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
