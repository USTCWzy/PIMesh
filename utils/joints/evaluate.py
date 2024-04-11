import os
import numpy as np
from core.evaluate import *

def joint_mapping(cal_mode):

    index = np.array([24, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8])

    if cal_mode == 'limb':
        cal_index = np.array([5, 6, 7, 8, 11, 12, 13, 14])
    elif cal_mode == 'limb_shoulder_nose':
        cal_index = np.array([0, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14])
    elif cal_mode == 'limb_shoulder':
        cal_index = np.array([3, 4, 5, 6, 7, 8, 11, 12, 13, 14])
    elif cal_mode == 'limb_shoulder_hip_nose':
        cal_index = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    elif cal_mode == 'limb_ear_shoulder_nose':
        cal_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14])
    elif cal_mode == 'limb_ear_shoulder':
        cal_index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14])

    return index[cal_index], cal_index

def joint_loss_calculation(file_path, label_file, joint_file_list):

    result = {}

    # for key in ['mpjpe', 'pck', 'accel', 'accel loss']:
    #     result[key] = []

    label = np.load(os.path.join(file_path, label_file))[:, :, :2]

    segments = np.load(os.path.join(file_path, 'Annotation', 'segment.npy'))


    for file in joint_file_list:

        mpjpe, accel, accel_loss = [], [], []

        data = np.load(os.path.join(file_path, file))[:, :, :2]

        name = file.replace('.npy', '')

        if len(segments.shape) > 1:
            for segment in segments:
                cal_label = label[segment[0]: segment[1]]
                cal_joint = data[segment[0]: segment[1]]

                mpjpe.extend(compute_errors(cal_label[:, :, :2], cal_joint[:, :, :2]))

                # pck = compute_pck(cal_label[:, :, :2], cal_joint[:, :, :2],
                #                   scale=np.sqrt(np.sum(np.square(
                #                       label[segment[0]: segment[1], 3, ] - label[segment[0]: segment[1], 10, :]), 1)))

                accel_loss.extend(compute_error_accel(cal_label[:, :, :2], cal_joint[:, :, :2]))

                accel.extend(compute_accel(cal_joint[:, :, :2]))

                # result['pck'].extend(pck)
                # result['mpjpe'].extend(loss)
                # result['accel'].extend(acc)
                # result['accel loss'].extend(acc_loss)

                # print(key, np.mean(loss), np.mean(pck), np.mean(acc_loss), np.mean(acc))

        else:

            cal_label = label[segments[0]: segments[1]]
            cal_joint = data[segments[0]: segments[1]]

            mpjpe.extend(compute_errors(cal_label[:, :, :2], cal_joint[:, :, :2]))

            # pck = compute_pck(cal_label[:, :, :2], cal_joint[:, :, :2],
            #                   scale=np.sqrt(np.sum(np.square(
            #                       label[segment[0]: segment[1], 3, ] - label[segment[0]: segment[1], 10, :]), 1)))

            accel_loss.extend(compute_error_accel(cal_label[:, :, :2], cal_joint[:, :, :2]))

            accel.extend(compute_accel(cal_joint[:, :, :2]))

            # result['pck'].extend(pck)
            # result['mpjpe'].extend(loss)
            # result['accel'].extend(acc)
            # result['accel loss'].extend(acc_loss)

        # accel_loss = compute_error_accel(label, data)
        # mpjpe = compute_errors(label, data)
        # accel_cal = compute_accel(data)

        if name == 'keypoints_meter':
            result['label'] = {}
            result['label']['mpjpe'] = mpjpe
            result['label']['accel_loss'] = accel_loss
            result['label']['accel'] = accel
        else:
            result[name[25:]] = {}
            result[name[25:]]['mpjpe'] = mpjpe
            result[name[25:]]['accel_loss'] = accel_loss
            result[name[25:]]['accel'] = accel

    return result


