from PIL import Image
import json
import math as M
import os, sys
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

import kaolin as kal

sys.path.append('..')
sys.path.append('../../..')

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from SMPL.smpl_torch_batch import SMPLModel

def star_draw_light(star_model, width=512, height=512, scale=1.0, cam_x=-0.6, cam_y=-1.2, cam_z=-1.5):
    '''
    可微的
    渲染点投影光照图
    return: [batch_size, width, height, 1]
    '''
    batch_size = star_model.shape[0]

    light_direction = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)

    #相机
    theta, phi, psi = 0, 0, 0

    cam_transform = torch.tensor(
        [[M.cos(theta)*M.cos(phi), M.sin(psi)*M.sin(theta)*M.cos(phi) - M.cos(psi)*M.sin(phi), M.cos(psi)*M.sin(theta)*M.cos(phi) + M.sin(psi)*M.sin(phi)],
        [M.cos(theta)*M.sin(phi),  M.sin(psi)*M.sin(theta)*M.sin(phi) + M.cos(psi)*M.cos(phi), M.cos(psi)*M.sin(theta)*M.sin(phi) - M.sin(psi)*M.cos(phi)],
        [ -M.sin(theta),           M.sin(psi)*M.cos(theta),                                    M.cos(psi)*M.cos(theta)],
        [ cam_x,  cam_y, cam_z]], device=DEVICE).repeat(batch_size, 1, 1)
    cam_proj = torch.tensor(
        [[ height / width],
         [ 1.0],
         [-1/scale]], device=DEVICE).repeat(batch_size, 1, 1)

    #star模型
    faces = torch.tensor(star_model.f.astype(np.int64), dtype=torch.int64, device=DEVICE)
    v_xzy = star_model[:, :, :]
    nb_faces = faces.shape[0]

    face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                v_xzy,
                faces, cam_proj, camera_transform=cam_transform
            )

    face_attributes = [
                face_normals.unsqueeze(2).repeat(1, 1, 3, 1),  #normals
            ]

    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                height, width, face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1]
            )
    face_normals_image = image_features[0]

    image = face_normals_image @ light_direction

    return image

    ## Display the rendered images
    f, axarr = plt.subplots(2, test_batch_size, figsize=(20, 20), squeeze=False)
    #f.subplots_adjust(top=0.99, bottom=0.79, left=0., right=1.4)
    f.suptitle('DIB-R rendering', fontsize=30)
    for i in range(test_batch_size):
        im = axarr[0][i].imshow(face_normals_image[i].cpu().detach())
        #f.colorbar(im, ax=axarr[0][i])
    for i in range(test_batch_size):
        im = axarr[1][i].imshow(image[i].cpu().detach())
        f.colorbar(im, ax=axarr[1][i])

def smpl_draw_light(smpl, faces, width=512, height=512, scale=1.0, cam_x=-0.6, cam_y=-1.2, cam_z=-1.5):
    '''
    可微的
    渲染点投影光照图
    return: [batch_size, width, height, 1]
    '''
    batch_size = smpl.shape[0]

    light_direction = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)

    #相机
    theta, phi, psi = 0, 0, 0

    cam_transform = torch.tensor(
        [[M.cos(theta)*M.cos(phi), M.sin(psi)*M.sin(theta)*M.cos(phi) - M.cos(psi)*M.sin(phi), M.cos(psi)*M.sin(theta)*M.cos(phi) + M.sin(psi)*M.sin(phi)],
        [M.cos(theta)*M.sin(phi),  M.sin(psi)*M.sin(theta)*M.sin(phi) + M.cos(psi)*M.cos(phi), M.cos(psi)*M.sin(theta)*M.sin(phi) - M.sin(psi)*M.cos(phi)],
        [ -M.sin(theta),           M.sin(psi)*M.cos(theta),                                    M.cos(psi)*M.cos(theta)],
        [ cam_x,  cam_y, cam_z]], device=DEVICE).repeat(batch_size, 1, 1)
    cam_proj = torch.tensor(
        [[ height / width],
         [ 1.0],
         [-1/scale]], device=DEVICE).repeat(batch_size, 1, 1)

    #star模型
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.int64, device=DEVICE)
    v_xzy = smpl[:, :, :]
    nb_faces = faces.shape[0]

    face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                v_xzy,
                faces, cam_proj, camera_transform=cam_transform
            )

    face_attributes = [
                face_normals.unsqueeze(2).repeat(1, 1, 3, 1),  #normals
            ]

    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                height, width, face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1]
            )
    face_normals_image = image_features[0]

    image = face_normals_image @ light_direction

    return image

    ## Display the rendered images
    f, axarr = plt.subplots(2, test_batch_size, figsize=(20, 20), squeeze=False)
    #f.subplots_adjust(top=0.99, bottom=0.79, left=0., right=1.4)
    f.suptitle('DIB-R rendering', fontsize=30)
    for i in range(test_batch_size):
        im = axarr[0][i].imshow(face_normals_image[i].cpu().detach())
        #f.colorbar(im, ax=axarr[0][i])
    for i in range(test_batch_size):
        im = axarr[1][i].imshow(image[i].cpu().detach())
        f.colorbar(im, ax=axarr[1][i])

def smpl_vis(path=r'/workspace/wzy1999/Public_Dataset/fyr_3Dpose_data', subject='03lgr',
             save_path=r'/workspace/wzy1999/3d_pose_estimation/pictures/SMPL/2023_2_7_star_2_smpl'):
    filename = os.path.join(path, subject, 'annotations_smpl.json')
    with open(filename, 'r') as f:
        annotations = json.loads(f.read())
        SMPL_model = SMPLModel(device=DEVICE, model_path='../SMPL/models/{}_model.pkl'.format(annotations[0]['gender']))
        batch_size = 200
        for i in trange(int(len(annotations) / batch_size), desc='STAR-SMPL转换'):
            smpl_verts, smpl_joints = SMPL_model(
                betas=torch.from_numpy(np.array([
                    element['smpl_betas'] for element in annotations[i * batch_size: (i + 1) * batch_size]
                ])).type(torch.float64).to(DEVICE),
                pose=torch.from_numpy(np.array([
                    element['smpl_poses'] for element in annotations[i * batch_size: (i + 1) * batch_size]
                ])).type(torch.float64).to(DEVICE),
                trans=torch.from_numpy(np.array([
                    element['smpl_trans'] for element in annotations[i * batch_size: (i + 1) * batch_size]
                ])).type(torch.float64).to(DEVICE)
            )
            faces = torch.tensor(SMPL_model.faces.astype(np.int64), dtype=torch.int64, device=DEVICE)
            smpl_verts = torch.tensor(smpl_verts, dtype=torch.float)
            image = smpl_draw_light(smpl_verts, SMPL_model.faces).cpu()
            for j, ele in enumerate(image):
                plt.imshow(ele)
                plt.savefig(os.path.join(
                    save_path, subject + '_{}.png').format(i * batch_size + j))
                plt.close()

def MMS_to_Bodies_at_Rest_Form(path):
    for subject in os.listdir(path):
        filename = os.path.join(path, subject, 'annotations_smpl.json')
        BR_data = {}
        with open(filename, 'r') as f:
            mms_data = json.loads(f.read())
            BR_data[b'body_mass'] = [mms_data[0]['weight_lbs'] for _ in range(len(mms_data))]
            BR_data[b'body_shape'] = [np.array(mms_data[i]['smpl_betas']) for i in range(len(mms_data))]
            BR_data[b'body_height'] = [mms_data[0]['height_in'] for _ in range(len(mms_data))]
            BR_data[b'bed_angle_deg'] = [0 for i in range(len(mms_data))]
            BR_data[b'root_xyz_shift'] = [np.array(mms_data[i]['smpl_trans']) for i in range(len(mms_data))]
            BR_data[b'images'] = [np.array(mms_data[i]['pressure']) for i in range(len(mms_data))]
            BR_data[b'joint_angles'] = [np.array(mms_data[i]['smpl_poses']).reshape(72) for i in range(len(mms_data))]
            BR_data[b'markers_xyz_m'] = [np.array(mms_data[i]['joint']).reshape(72) for i in range(len(mms_data))]
            BR_data[b'mesh_contact'] = [np.where(np.array(mms_data[i]['pressure']).reshape(56, 40) > 0) for i in range(len(mms_data))]
            pkl.dump(BR_data, open(os.path.join(path, subject, 'annotations_br_form.p'), 'wb'))

if __name__ == '__main__':
    MMS_to_Bodies_at_Rest_Form(r'/workspace/wzy1999/Public_Dataset/fyr_3Dpose_data')