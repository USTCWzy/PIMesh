import os
import time
from tqdm import tqdm
import numpy as np
from utils.videos.convert import *

def video_segment(key_point_path, video_path, video_name, output_path, fps=25):
    keypoint_num = 21
    input_fps = 15

    video_file_path = os.path.join(video_path, video_name)
    key_point_path = os.path.join(key_point_path, 'keypoints.txt')

    total_time = time.time()

    image_folder, num_frames, img_shape = video_to_images(video_file_path, return_info=True)
    print(f'Input video number of frames {num_frames}')

    key_points = np.zeros((num_frames, keypoint_num * 3))
    video_timestamps = np.zeros(num_frames)
    key_points_flag = np.zeros(num_frames)
    with open(key_point_path, 'r') as f:
        data = f.readline()
        while data:
            data = list(map(float, data[:-1].split('\t')[:-1]))
            if len(data) == 77:
                #import pdb;pdb.set_trace()
                key_points[int(data[0]) - 1][:15 * 3] = np.array(data[2:])[:15 * 3]
                key_points[int(data[0]) - 1][15 * 3:] = np.array(data[2:])[19 * 3:]
                video_timestamps[int(data[0]) - 1] = data[1]
                key_points_flag[int(data[0]) - 1] = 1
            data = f.readline()
    key_points = key_points.reshape((num_frames, keypoint_num, 3))

    key_points_detection_flag = np.sum(key_points[:, :, :2], axis=2) > 10
    key_points_human_detection_flag = list(map(int, np.sum(key_points_detection_flag, axis=1) > 8))

    video_raw_seg, video_seg, start_idx, end_idx, merge = [], [], 0, 0, True
    key_points_human_detection_flag = [0] + key_points_human_detection_flag + [0]
    for i in range(len(key_points_human_detection_flag) - 1):
        if not key_points_human_detection_flag[i] and key_points_human_detection_flag[i + 1]:
            start_idx = i
        if key_points_human_detection_flag[i] and not key_points_human_detection_flag[i + 1]:
            end_idx = i
            if end_idx - start_idx > 10 * input_fps:
                video_raw_seg.append([start_idx - 1, end_idx - 1])

    video_seg = video_raw_seg.copy()
    while merge:
        for idx in range(len(video_seg) - 1):
            if video_seg[idx + 1][0] - video_seg[idx][1] < 3 * input_fps:
                video_seg[idx][1] = video_seg[idx + 1][1]
                video_seg.remove(video_seg[idx + 1])
                break
            if idx == len(video_seg) - 2:
                merge = False
    # import pdb;pdb.set_trace()
    print('len of video_seg:', len(video_seg))
    for idx, seg_idx in enumerate(video_seg):
        # images_to_video_w_seg(img_folder=image_folder, output_vid_file=os.path.join(output_path, f'{idx}.mp4'), fps=fps,
        #                       start_idx=seg_idx[0], end_idx=seg_idx[1])
        if input_fps == 15:
            start_int_time = seg_idx[0] // input_fps
            start_dec_time = seg_idx[0] / input_fps - start_int_time
            if start_dec_time == 0:
                start_time = start_int_time
            elif start_dec_time < 0.2:
                start_time = start_int_time + 0.2
            elif start_dec_time < 0.4:
                start_time = start_int_time + 0.4
            elif start_dec_time < 0.6:
                start_time = start_int_time + 0.6
            elif start_dec_time < 0.8:
                start_time = start_int_time + 0.8
            else:
                start_time = start_int_time + 1.0
        else:
            start_time = seg_idx[0] / input_fps

        if input_fps == 15:
            end_int_time = (seg_idx[1] - int(start_time * input_fps)) // input_fps
            end_dec_time = (seg_idx[1] - int(start_time * input_fps)) / input_fps - end_int_time
            if end_dec_time == 0:
                end_time = end_int_time
            elif end_dec_time < 0.2:
                end_time = end_int_time
            elif end_dec_time < 0.4:
                end_time = end_int_time + 0.2
            elif end_dec_time < 0.6:
                end_time = end_int_time + 0.4
            elif end_dec_time < 0.8:
                end_time = end_int_time + 0.6
            else:
                end_time = end_int_time + 0.8
        else:
            end_time = (seg_idx[1] - seg_idx[0]) / input_fps
        print('raw index: ', seg_idx)
        print('processed index: ', [int(start_time * input_fps), int(start_time * input_fps + end_time * input_fps)])
        video_trunc(video_file_path, os.path.join(output_path, 'videos', f'{idx}.mp4'), start_time, end_time, fps)
        np.save(os.path.join(output_path, 'keypoints', f'keypoints_{idx}.npy'),
                key_points[int(start_time * input_fps): int(start_time * input_fps + end_time * input_fps)])
        np.save(os.path.join(output_path, 'timestamp', f'timestamp_{idx}.npy'),
                video_timestamps[int(start_time * input_fps): int(start_time * input_fps + end_time * input_fps)])
    np.save(os.path.join(video_path, 'segment.npy'), np.array(video_seg))
    total_time = time.time() - total_time
    print('total run time: ', total_time)