import sys
import os

import os.path as osp

import time
import yaml
import smplx
import torch
import pickle
import logging
import numpy as np

from torch.utils.data import DataLoader
from utils.vis.smpl_vis import batch_optimization_render

from dataset.InBedPressureDataset import InBedPressureDataset

from config.cmd_train_parser import parser_train_config

dataset_idx_mapping = [
    0, 1, 2, 3, 4, 5,
    1, 2, 3, 4,
    1, 2, 3, 4, 5,
    1, 2, 3, 4, 5, 6,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4
]

def test_vis(args, result_path):

    cfgs = {
            'dataset_path': args.dataset_path,
            'save_dataset_path': '',
            'dataset_mode': args.exp_mode,
            'curr_fold': args.curr_fold,
            'seqlen': args.seqlen,
            'overlap': args.overlap,
            'normalize': True,
            'img_size': args.pi_img_size
        }

    # test_set = InBedPressureDataset(
    #     cfgs,
    #     mode='test'
    # )
    # test_loader = DataLoader(
    #     dataset=test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=2
    # )

    results = dict(np.load(os.path.join(result_path, 'test.npz'), allow_pickle=True))

    date = results['info'].item()['date']
    name = results['info'].item()['name']
    idx = results['info'].item()['idx']
    corner = np.array(results['info'].item()['corner'])
    sensor_position = np.array(results['info'].item()['sensor_position'])

    segments = results['segments']
    db_segments = results['db_segmemts']

    body_model = smplx.create('data/models', model_type='smpl', gender=args.smpl_gender).to('cpu')
    body_model.eval()

    for i, (segment, db_segment) in enumerate(zip(segments, db_segments)):
        print(name[i], idx[i], segment)
        assert(segment[1] - segment[0] == db_segment[1] - db_segment[0])

        segment_index = np.array([i for i in range(segment[0], segment[1])])
        db_segment_index = np.array([i for i in range(db_segment[0], db_segment[1])])

        rgb_path = rf'{name[i]}/Group{dataset_idx_mapping[idx[i]]}'

        results['theta'][segment[0]:segment[1], :2] -= ([0, 55 * 0.0311] - (corner[segment[0]:segment[1], :] - sensor_position[i][0])) * [-1, 1]

        save_path = '/' + args.exp_mode
        if args.exp_mode == 'unseen_subject':
            save_path += '_'+ str(args.curr_fold)


        for j in range(0, segment[1] - segment[0], 128):

            model_output = body_model(
                betas=torch.tensor(results['theta'][segment_index[j: j + 128], 75:], dtype=torch.float32),
                global_orient=torch.tensor(results['theta'][segment_index[j: j + 128], 3:6], dtype=torch.float32),
                body_pose=torch.tensor(results['theta'][segment_index[j: j + 128], 6:75], dtype=torch.float32),
                transl=torch.tensor(results['theta'][segment_index[j: j + 128], :3], dtype=torch.float32)
            )

            batch_optimization_render(
                rgb_path=rgb_path,
                curr_frame_idx=[db_segment_index[j: j + 128][0], db_segment_index[j: j + 128][-1]],
                verts=model_output.vertices.detach().numpy(),
                faces=body_model.faces,
                dataset_path=r'/workspace/wzy1999/Public_Dataset/wzy_dataset',
                output_path=r'/workspace/wzy1999/Public_Dataset/wzy_final_results_vis_seq_1' + save_path,
                show_sideView=True
            )



    print(1)

if __name__ == '__main__':
    args = parser_train_config()

    path = '/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/results/PIMesh_unseen_group_resnet50_trans_spin_0.005_256_1_0.25_10.0_0_100.0_100_300.0_50.0_50.0_50.0_20.0_0.0_cosine_test'

    # j = 3
    # path = f'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/results/PIMesh_{j}_resnet50_trans_spin_0.005_16_16_0.25_10.0_0_100.0_100_300.0_50.0_50.0_50.0_20.0_0.0_cosine_test'

    ckp_name = path.split('/')[-1]
    params_list = ckp_name.split('_')
    # FOR RESNET
    if 'unseen_group' in ckp_name:

        args.exp_mode = 'unseen_group'
        args.encoder = params_list[3]
        args.temp_encoder = params_list[4]
        args.seqlen = int(params_list[8])
        args.batch_size = int(params_list[7])
        args.cosine = 1

        test_vis(args, path)
    else:
        args.exp_mode = 'unseen_subject'
        args.encoder = params_list[2]
        args.temp_encoder = params_list[3]
        args.seqlen = int(params_list[7])
        args.batch_size = int(params_list[6])
        args.cosine = 1
        args.curr_fold = int(params_list[1])

        test_vis(args, path)