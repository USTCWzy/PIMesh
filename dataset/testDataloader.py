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
    'wq': [1, 6],
    'lgr': [6, 10],
    'wyc': [10, 15],
    'zyk': [15, 21],
    'nmt': [21, 25],
    'wyx': [25, 29],
    'lz': [29, 33],
    'twj': [33, 37],
    'xft': [37, 41],
}

three_fold = [
    ['wyc', 'lgr', 'wyx'],
    ['wq', 'nmt', 'twj'],
    ['xft', 'zyk', 'lz'],
]


class TestLoader(Dataset):
    def __init__(self, cfgs, mode='train'):

        self.cfgs = cfgs
        self.test_dataset_name = cfgs['dataset_path']
        self.save_dataset_path = cfgs['save_dataset_path']
        self.seqlen = cfgs['seqlen']
        self.overlap = cfgs['overlap']
        self.mid_frame = int(self.seqlen / 2)
        self.stride = int(self.seqlen * (1 - self.overlap) + 0.5)
        self.normalize = cfgs['normalize']
        self.img_size = cfgs['img_size']

        self.segments = []
        self.db_segmemts = []

        self.data_len = 0

        if self.test_dataset_name.endswith('.csv'):
            self.data = pd.read_csv(self.test_dataset_name, header=None).values
            self.data = self.data[:, 3:].reshape(-1, 56, 40)
        elif self.test_dataset_name.endswith('.npy'):
            self.data = np.load(self.test_name)

        self.segments = np.array([[0, self.data.shape[0]]])
        self.db_segmemts = np.array([[0, self.data.shape[0]]])
        self.data_len = self.data.shape[0]

        self.video_index_list, self.sample_lens = self.video_seg_window()

        pressure = np.zeros((self.data.shape[0], self.img_size[0], self.img_size[1])).astype(np.float32)
        hor_margin = (self.img_size[1] - self.data.shape[2]) // 2
        ver_margin = (self.img_size[0] - self.data.shape[1]) // 2
        pressure[:, ver_margin: ver_margin + self.data.shape[1], hor_margin: hor_margin + self.data.shape[2]] \
            = self.data

        if self.normalize:

            pressure[pressure > 512] = 512

            pressure = pressure / (512 - np.min(pressure))


        self.data = pressure.copy()




    def __len__(self):
        return len(self.video_index_list)

    def __getitem__(self, index):

        start_idx, end_idx, len_judge = self.video_index_list[index]

        return {
            'curr_frame_idx': torch.tensor([i for i in range(start_idx, end_idx)]),
            'images': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data)
            )
        }

    def get_sequence(self, start_index, end_index, len_judge, data):
        if len_judge:
            return data[start_index: end_index]
        else:
            index_list = [i for i in range(start_index, end_index)] + [end_index - 1 for _ in
                                                                       range(self.seqlen - (end_index - start_index))]
            return data[index_list]

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

    def get_segments(self):
        return self.segments

    def get_data_len(self):
        return self.data_len









