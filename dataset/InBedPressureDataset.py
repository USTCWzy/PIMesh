import os
import h5py
import torch
import numpy as np
import pandas as pd
import torchvision as tv
from torch.utils.data import Dataset

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

name_group_map = {
    '1': [1, 6],
    '2': [6, 10],
    '3': [10, 15],
    '4': [15, 21],
    '5': [21, 25],
    '6': [25, 29],
    '7': [29, 33],
    '8': [33, 37],
    '9': [37, 41],
}

three_fold = [
    ['3', '2', '6'],
    ['1', '5', '8'],
    ['9', '4', '7'],
]


class InBedPressureDataset(Dataset):
    def __init__(self, cfgs, mode='train'):

        self.cfgs = cfgs
        self.dataset_name = cfgs['dataset_path']
        self.save_dataset_path = cfgs['save_dataset_path']
        self.seqlen = cfgs['seqlen']
        self.overlap = cfgs['overlap']
        self.mid_frame = int(self.seqlen / 2)
        self.stride = int(self.seqlen * (1 - self.overlap) + 0.5)
        self.model_type = cfgs['dataset_mode']
        self.curr_fold = cfgs['curr_fold']
        self.mode = mode
        self.normalize = cfgs['normalize']
        self.img_size = cfgs['img_size']

        self.segments = []
        self.db_segmemts = []

        self.data_len = 0

        self.data = {}
        self.info = {
            'date': [],
            'name': [],
            'idx': [],
            'corner': [],
            'sensor_position': []
        }

        if self.model_type == 'unseen_group':
            if self.mode == 'train':
                for name in name_group_map:
                    if name == '3':
                        for idx in [10, 11, 14]:
                            print(f'load train dataset: {idx}')
                            self.load_db(idx)
                    else:
                        for idx in range(name_group_map[name][0], name_group_map[name][1])[:-2]:
                            print(f'load train dataset: {idx}')
                            self.load_db(idx)

            elif self.mode == 'eval':
                for name in name_group_map:
                    if name == '3':
                        idx = name_group_map[name][1] - 3
                    else:
                        idx = name_group_map[name][1] - 2
                    print(f'load val dataset: {idx}')
                    self.load_db(idx)
            else:
                for name in name_group_map:
                    if name == '3':
                        idx = name_group_map[name][1] - 2
                    else:
                        idx = name_group_map[name][1] - 1
                    print(f'load test dataset: {idx}')
                    self.load_db(idx)

        elif self.model_type == 'unseen_subject':
            if self.mode == 'train':
                for fold, name_list in enumerate(three_fold):
                    if fold != self.curr_fold - 1:
                        for name in three_fold[fold]:
                            for idx in range(name_group_map[name][0], name_group_map[name][1]):
                                print(f'load train dataset: {idx}')
                                self.load_db(idx)

            elif self.mode == 'eval':
                for name in three_fold[self.curr_fold - 1]:
                    for idx in range(name_group_map[name][0], name_group_map[name][1]):
                        print(f'load val dataset: {idx}')
                        self.load_db(idx)

        self.video_index_list, self.sample_lens = self.video_seg_window()

        pressure = np.zeros((self.data['pressure'].shape[0], self.img_size[0], self.img_size[1])).astype(np.float32)
        hor_margin = (self.img_size[1] - self.data['pressure'].shape[2]) // 2
        ver_margin = (self.img_size[0] - self.data['pressure'].shape[1]) // 2
        pressure[:, ver_margin: ver_margin + self.data['pressure'].shape[1], hor_margin: hor_margin + self.data['pressure'].shape[2]] \
            = self.data['pressure']

        if self.normalize:

            '''mean_value = np.mean(self.data)
            max_value = np.max(self.data)
            std_value = np.std(self.data)

            self.data = self.data / max_value
            self.data = (self.data - mean_value) / std_value
            base_noise = mean_value / std_value'''

            pressure[pressure > 512] = 512

            pressure = pressure / (512 - np.min(pressure))


        self.data['pressure'] = pressure.copy()


    def load_db(self, idx):
        db = dict(np.load(os.path.join(self.dataset_name, f'data_{idx}.npz'),
                     allow_pickle=True))

        sensor_position = db['infer_sensor_position']
        segments = db['segments']

        data = {
            'pressure': db['pressure'],
            'binary_pressure': db['binary_pressure'],
            # 'keypoints_pi': (sensor_position[0] + np.array([0, 55 * 0.0311]) - db['keypoints_meter_smooth']) / np.array(
            #             [-0.0195, 0.0311]),
            'keypoints_pi': db['keypoints_meter_smooth'] / np.array([0.0195, 0.0311]),
            'betas': db['label_betas'],
            'pose': db['label_pose'],
            'trans': db['label_trans'],
            # 'verts': db['label_verts'],
            # 'keypoints_3d': db['label_kp_3d']
            'verts': db['label_verts'],
            'keypoints_3d': db['label_joints'][:, :25, :]
        }

        for segment in segments:

            self.info['name'].append(db['name'])
            self.info['date'].append(db['date'])
            self.info['sensor_position'].append(db['infer_sensor_position'])
            self.info['corner'].extend(db['bed_corner_shift'][segment[0]: segment[1]])
            self.info['idx'].append(idx)

            if not len(self.data):
                for key in data.keys():
                    self.data[key] = data[key][segment[0]: segment[1]]
                self.segments.append(np.array(segment) - segment[0] + self.data_len)
                self.db_segmemts.append(segment)
                self.data_len += segment[1] - segment[0]
            else:
                for key in data.keys():
                    self.data[key] = np.concatenate([self.data[key], data[key][segment[0]: segment[1]]], axis=0)
                self.segments.append(np.array(segment) - segment[0] + self.data_len)
                self.db_segmemts.append(segment)
                self.data_len += segment[1] - segment[0]

    def __len__(self):
        return len(self.video_index_list)

    def __getitem__(self, index):

        start_idx, end_idx, len_judge = self.video_index_list[index]

        return {
            'curr_frame_idx': torch.tensor([i for i in range(start_idx, end_idx)]),
            'images': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['pressure'])
            ),
            'pressure_binary': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['binary_pressure'])
            ),
            'gt_keypoints_2d': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['keypoints_pi'])
            ),
            'betas': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['betas'])
            ),
            'pose': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['pose'])
            ),
            'trans': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['trans'])
            ),
            'verts': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['verts'])
            ),
            'gt_keypoints_3d': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['keypoints_3d'])
            ),
        }

    def get_sequence(self, start_index, end_index, len_judge, data):
        if len_judge:
            return data[start_index: end_index]
        else:
            index_list = [i for i in range(start_index, end_index)] + [end_index - 1 for _ in
                                                                       range(self.seqlen - (end_index - start_index))]
            return data[index_list]

    def get_single_item(self, index):
        pass

    def video_seg_window(self):

        index_list = []

        for segment in self.segments:

            if segment[1] - segment[0] < self.seqlen:
                index_list.append([segment[0], segment[1], 0])
            else:
                for i in range(segment[0], segment[1], self.stride):
                    if i + self.seqlen >= segment[1]:
                        index_list.append([segment[1] - self.seqlen, segment[1], 1])
                    else:
                        index_list.append([i, i + self.seqlen, 1])

        return index_list, len(index_list)

    def get_curr_image_path(self):
        return f'{self.date}{self.name}/{self.date}_{self.name}{dataset_idx_mapping[self.curr_idx]}'

    def save_opt_results(self, curr_frame_idx, pose, shape, trans):

        start_index, end_index, len_judge = \
            curr_frame_idx[0], curr_frame_idx[1], curr_frame_idx[2]

        self.db['label_pose'][start_index: end_index] = pose[: end_index - start_index]
        self.db['label_betas'][start_index: end_index] = shape[: end_index - start_index]
        self.db['label_trans'][start_index: end_index] = trans[: end_index - start_index]

    def save_db(self):
        np.savez(
            os.path.join(self.cfgs['save_dataset_path'], f'data_{self.curr_idx}.npz'),
            **self.db
        )

    def get_segments(self):
        return self.segments

    def get_data_len(self):
        return self.data_len









