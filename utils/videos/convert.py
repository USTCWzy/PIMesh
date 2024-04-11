import os
import cv2
import subprocess
import os.path as osp

def video_to_images(vid_file, img_folder=None, return_info=False, note=''):
    # copied from vibe
    if img_folder is None:
        img_folder = osp.join('tmp/', osp.basename(vid_file).replace('.mp4', ''))

    # if osp.isdir(img_folder) and len(os.listdir(img_folder)) > 0:
    #     img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape
    #     return img_folder, len(os.listdir(img_folder)), img_shape

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def images_to_video(img_folder, output_vid_file):
    # copied from vibe
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error',
        output_vid_file
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def images_to_video_w_seg(img_folder, output_vid_file, fps=25, start_idx=None, end_idx=None):
    # this codes have some bugs and i can't solve.
    os.makedirs(img_folder, exist_ok=True)

    segment = f"select=between(n\\,{start_idx}\\,{end_idx})"

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-r', f'{fps}',
        "-vf", segment, output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def video_trunc(input_video, output_video, start_time, seconds, fps):
    command = [
        'ffmpeg', '-y', '-threads', '16', '-ss', f'{start_time}', '-t', f'{seconds}', '-i', input_video,
        '-c', 'copy', '-r', f'{fps}', '-v', 'error',output_video
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def video_fps(input_video, output_video, fps):
    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', input_video, '-c:v', 'libx264',
        '-v', 'error', '-r', f'{fps}', output_video
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def rm_mid_video(output_path):
    command = [
        'rm', os.path.join(output_path, '*_mid.mp4')
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

