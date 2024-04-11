import os
import cv2
import time
from tqdm import tqdm
import numpy as np
from utils.videos.convert import *
from utils.joints.smooth import create_smooth
from core.evaluate import *

BODY_25_color = np.array([
        [255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0],
        [255, 255, 0], [204, 255, 0], [153, 255, 0], [102, 255, 0], [51, 255, 0],
        [0, 255, 0], [0, 255, 51], [0, 255, 102], [0, 255, 153], [0, 255, 204],
        [0, 255, 255], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 53, 255],
        [0, 0, 255], [53, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255],
        [255, 0, 255]])

BODY_25_pairs = np.array([
        [1, 8], [1, 2], [1, 5], [2, 3], [3, 4],
        [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
        [8, 12], [12, 13], [13, 14], [1, 0], [14, 15],
        [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]
                        ])



def joint_in_image_vis(image_file_list, image_index_list, joints, key_points_flag_list, key_points_detection_flag_list,
                       tmp_image_openpose_folder, keypoint_num=21, cam_offset=0):
    for frame_idx in tqdm(image_index_list):
        img = cv2.imread(image_file_list[frame_idx])
        if len(joints) == 1:
            joint = joints[0]
            key_points_flag = key_points_flag_list[0]
            key_points_detection_flag = key_points_detection_flag_list[0]
            if key_points_flag[frame_idx]:
                for kp_idx in range(keypoint_num):
                    if key_points_detection_flag[frame_idx, kp_idx]:
                        # xy = tuple(cam[frame_idx-cam_offset, kp_idx, :2].astype(int))
                        xy = tuple(joint[frame_idx - cam_offset, kp_idx, :2].astype(int))

                        color = BODY_25_color[kp_idx]
                        color = tuple([int(x) for x in color])
                        cv2.circle(img, xy, 5, tuple(color), 5)

                for pair_idx in range(20):
                    pair = BODY_25_pairs[pair_idx]
                    idx_1, idx_2 = pair[0], pair[1]
                    if key_points_detection_flag[frame_idx, idx_1] and key_points_detection_flag[frame_idx, idx_2]:
                        color = BODY_25_color[pair_idx]
                        color = tuple([int(x) for x in color])

                        p1 = tuple([int(x) for x in joint[frame_idx - cam_offset, idx_1, :2]])
                        p2 = tuple([int(x) for x in joint[frame_idx - cam_offset, idx_2, :2]])

                        cv2.line(img, p1, p2, tuple(color), 5)
                cv2.imwrite(os.path.join(tmp_image_openpose_folder, f'{frame_idx:06d}.png'), img)
        else:
            image_horizontal_list = []
            for (joint, key_points_flag, key_points_detection_flag) in \
                    zip(joints, key_points_flag_list, key_points_detection_flag_list):
                img_c = img.copy()
                if key_points_flag[frame_idx]:
                    for kp_idx in range(keypoint_num):
                        if key_points_detection_flag[frame_idx, kp_idx]:
                            # xy = tuple(cam[frame_idx-cam_offset, kp_idx, :2].astype(int))
                            xy = tuple(joint[frame_idx - cam_offset, kp_idx, :2].astype(int))

                            color = BODY_25_color[kp_idx]
                            color = tuple([int(x) for x in color])
                            cv2.circle(img_c, xy, 5, tuple(color), 5)

                    for pair_idx in range(20):
                        pair = BODY_25_pairs[pair_idx]
                        idx_1, idx_2 = pair[0], pair[1]
                        if key_points_detection_flag[frame_idx, idx_1] and key_points_detection_flag[
                            frame_idx, idx_2]:
                            color = BODY_25_color[pair_idx]
                            color = tuple([int(x) for x in color])

                            p1 = tuple([int(x) for x in joint[frame_idx - cam_offset, idx_1, :2]])
                            p2 = tuple([int(x) for x in joint[frame_idx - cam_offset, idx_2, :2]])

                            cv2.line(img_c, p1, p2, tuple(color), 5)
                image_horizontal_list.append(img_c)
                hor_img = cv2.hconcat(image_horizontal_list)
                cv2.imwrite(os.path.join(tmp_image_openpose_folder, f'{frame_idx:06d}.png'), hor_img)

def horizontal_video_vis(video_list, output_dir, output_name):
    src_image_list = []
    total_time = time.time()
    for src_video_file in video_list:
        # import pdb;pdb.set_trace()
        image_folder, num_frames, img_shape = video_to_images(src_video_file, return_info=True)
        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        src_image_list.append(image_file_names)

    tmp_image_openpose_folder = os.path.join('tmp/', output_name.replace('.', '_') + 'concat')
    if not os.path.isdir(tmp_image_openpose_folder):
        os.makedirs(tmp_image_openpose_folder)

    for frame_idx in tqdm(range(len(image_file_names))):
        img = cv2.hconcat([cv2.imread(image_file_names[frame_idx]) for image_file_names in src_image_list])
        cv2.imwrite(os.path.join(tmp_image_openpose_folder, f'{frame_idx:06d}.png'), img)

    images_to_video(img_folder=tmp_image_openpose_folder, output_vid_file=os.path.join(output_dir, output_name))

    total_time = time.time() - total_time
    print('total run time: ', total_time)

def image_joint_comparision(image_folder, joint_list):
    keypoint_num = 21
    cam_offset = 0

    tmp_image_openpose_folder = os.path.join('tmp/', f'comparision')
    if not os.path.isdir(tmp_image_openpose_folder):
        os.makedirs(tmp_image_openpose_folder)

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    missing_index_list = np.argwhere((joint_list[-1][:, :-6, 2] > 0.6001).all(axis=1) == False).squeeze()

    key_points_detection_flag = [
        np.sum(joint[:, :, :2], axis=2) > 10 for joint in joint_list]

    joint_in_image_vis(
        image_file_names, missing_index_list, joint_list,
        [np.ones(joint_list[-1].shape[0]) for _ in range(len(joint_list))],
        key_points_detection_flag, tmp_image_openpose_folder=tmp_image_openpose_folder
    )

def openpose_video_vis(key_point_path, video_path, video_name, keypoints_name, output_path, args, note='', fps=25,
                       start_idx=0, end_index=None):
    keypoint_num = 21
    cam_offset = 0

    BODY_25_color[:, [0, 2]] = BODY_25_color[:, [2, 0]]

    video_file_path = os.path.join(video_path, video_name)
    video_out_path = os.path.join(output_path, f'{video_name.replace(".mp4", "")}_openpose_vis_{note}.mp4')
    key_point_path = os.path.join(key_point_path, keypoints_name)

    total_time = time.time()

    image_folder, num_frames, img_shape = video_to_images(video_file_path, return_info=True)
    print(f'Input video number of frames {num_frames}')

    tmp_image_openpose_folder = os.path.join('tmp/', f'{video_name.replace(".mp4", "")}_openpose_vis_{note}')
    if not os.path.isdir(tmp_image_openpose_folder):
        os.makedirs(tmp_image_openpose_folder)

    if key_point_path.endswith('.txt'):
        key_points_flag = np.zeros(num_frames)
        key_points = np.zeros((num_frames, keypoint_num * 3))
        with open(key_point_path, 'r') as f:
            data = f.readline()
            while data:
                data = list(map(float, data[:-1].split('\t')[:-1]))
                if len(data) == 77:
                    # import pdb;pdb.set_trace()
                    key_points[int(data[0]) - 1][: 14 * 3] = np.array(data[2:])[: 14 * 3]
                    key_points[int(data[0]) - 1][14 * 3:] = np.array(data[2:])[18 * 3:]
                    key_points_flag[int(data[0]) - 1] = 1
                data = f.readline()
        key_points = key_points.reshape((num_frames, keypoint_num, 3))
    elif key_point_path.endswith('.npy'):
        key_points_flag = np.ones(num_frames)
        key_points = np.load(key_point_path)

    smooth = create_smooth(args)(args, key_points)

    key_points_detection_flag = np.sum(smooth[:, :, :2], axis=2) > 10

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])
    if end_index is None:
        end_index = len(image_file_names)

    joint_in_image_vis(image_file_names, [i for i in range(start_idx, end_index)], [smooth], [key_points_flag],
                       [key_points_detection_flag], tmp_image_openpose_folder)

    images_to_video(img_folder=tmp_image_openpose_folder, output_vid_file=video_out_path)

    total_time = time.time() - total_time
    print('total run time: ', total_time)

    np.save(os.path.join(video_path, 'smooth_' + note + '.npy'), smooth)
    print(compute_errors(key_points[:, :, :2], smooth[:, :, :2]))
