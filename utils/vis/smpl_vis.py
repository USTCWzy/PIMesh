import matplotlib.pyplot as plt
import os, cv2

import numpy as np
from tqdm import tqdm
from .renderer_pyrd import Renderer
def plot_3d(verts, verts_label, mask1=None, mask2=None, rgb=None, keypoints_list=None, save_path=None, save_name=None):
    points = verts

    color_list = ['blue', 'red']

    fig = plt.figure(figsize=(20, 10))
    if rgb is None:
        ax = fig.add_subplot(121, projection='3d')

        ax.scatter(points[mask2][:, 0], points[mask2][:, 1], points[mask2][:, 2],
                   s=10,
                   color='blue',
                   linewidth=0,
                   alpha=1,
                   marker=".")
        ax.scatter(points[mask1][:, 0], points[mask1][:, 1], points[mask1][:, 2],
                   s=10,
                   color='red',
                   linewidth=0,
                   alpha=1,
                   marker=".")

        #     ax.scatter(points[:,0], points[:,1], points[:,2],
        #            s=10,
        #            color='red',
        #            linewidth=0,
        #            alpha=1,
        #            marker=".")

        plt.title('Point Cloud', y=0.1)
        # ax.axis('scaled')  # {equal, scaled}
        ax.view_init(0, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax1 = fig.add_subplot(122, projection='3d')

        ax1.scatter(verts_label[mask2][:, 0], verts_label[mask2][:, 1], verts_label[mask2][:, 2],
                   s=10,
                   color='blue',
                   linewidth=0,
                   alpha=1,
                   marker=".")
        ax1.scatter(verts_label[mask1][:, 0], verts_label[mask1][:, 1], verts_label[mask1][:, 2],
                   s=10,
                   color='red',
                   linewidth=0,
                   alpha=1,
                   marker=".")

        #     ax.scatter(points[:,0], points[:,1], points[:,2],
        #            s=10,
        #            color='red',
        #            linewidth=0,
        #            alpha=1,
        #            marker=".")

        plt.title('Point Cloud', y=0.1)
        # ax.axis('scaled')  # {equal, scaled}
        ax1.view_init(90, 0)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
    else:
        ax = fig.add_subplot(121, projection='3d')

        ax.scatter(points[mask2][:, 0], points[mask2][:, 1], points[mask2][:, 2],
                   s=10,
                   color='blue',
                   linewidth=0,
                   alpha=1,
                   marker=".")
        ax.scatter(points[mask1][:, 0], points[mask1][:, 1], points[mask1][:, 2],
                   s=10,
                   color='red',
                   linewidth=0,
                   alpha=1,
                   marker=".")

        #     ax.scatter(points[:,0], points[:,1], points[:,2],
        #            s=10,
        #            color='red',
        #            linewidth=0,
        #            alpha=1,
        #            marker=".")

        plt.title('Point Cloud', y=0.1)
        # ax.axis('scaled')  # {equal, scaled}
        ax.view_init(0, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax = fig.add_subplot(122)

        ax.imshow(rgb[:, :, ::-1])

        if keypoints_list is not None:
            for keypoints in keypoints_list:
                ax.scatter(keypoints[:, 0], keypoints[:, 1])
    plt.show()
    # plt.savefig(os.path.join(save_path, save_name + '.png'))


def batch_optimization_render(rgb_path, curr_frame_idx, verts, faces, dataset_path=r'/workspace/wzy1999/Public_Dataset/wzy_dataset',
                              output_path=r'/workspace/wzy1999/Public_Dataset/wzy_opt_render_vis', show_sideView=True):
    input_path = os.path.join(dataset_path, rgb_path, 'rgb_image')

    orig_img_bgr_all = [
        cv2.imread(os.path.join(input_path, f'{idx + 1:06d}.png')) for idx in range(curr_frame_idx[0], curr_frame_idx[1])
    ]

    output_image_path = os.path.join(output_path, rgb_path)
    os.makedirs(output_image_path, exist_ok=True)

    img_h, img_w, _ = orig_img_bgr_all[0].shape
    focal_length = 941

    verts += [0, 0, 1.66]

    for img_idx in range(curr_frame_idx[0], curr_frame_idx[1]):
        orig_img_bgr = orig_img_bgr_all[img_idx - curr_frame_idx[0]]
        chosen_vert_arr = verts[img_idx - curr_frame_idx[0]]


        # setup renderer for visualization
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=faces,
                            same_mesh_color=True)
        front_view = renderer.render_front_view(chosen_vert_arr,
                                                bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
        front_view_1 = renderer.render_front_view(chosen_vert_arr,
                                                bg_img_rgb=np.ones_like(orig_img_bgr[:, :, ::-1]) * 127)


        # save rendering results


        if show_sideView:
            side_view_img = renderer.render_side_view(verts[img_idx - curr_frame_idx[0]:
                                                            img_idx - curr_frame_idx[0] + 1])
            # cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

        if show_sideView:
            cv2.imwrite(os.path.join(output_image_path, f'{img_idx:06d}_both.png'),
                        cv2.hconcat([front_view, front_view_1, side_view_img])[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(output_image_path, f'{img_idx:06d}.png'), front_view[:, :, ::-1])
        # plt.imshow(cv2.hconcat([front_view, front_view_1, side_view_img])[:, :, ::-1]);plt.show()

        # delete the renderer for preparing a new one
        renderer.delete()
