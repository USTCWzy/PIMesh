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

from dataset.InBedPressureDataset import InBedPressureDataset

from model.PIMesh import PIMeshNet

from core.loss import HPSPILoss
from core.trainer import Trainer

from utils.geometry.camera import PerspectiveCamera
from utils.optimizers.optim_factory import create_optimizer
from utils.optimizers.optim_factory_mae import LayerDecayValueAssigner, create_optimizer
from utils.others.utils import NoteGeneration, setup_seed
from utils.others.loss_record import updateLoss
from config.cmd_train_parser import parser_train_config

from pytorch3d.renderer import (
                                RasterizationSettings, MeshRenderer,
                                MeshRasterizer,
                                SoftSilhouetteShader,
                                )
from pytorch3d.renderer.cameras import OrthographicCameras

def main(args):

    start = time.time()

    setup_seed(42)

    logging_path = os.path.join(args.logging_path, NoteGeneration(args))
    checkpoints_path = os.path.join(args.checkpoints_path, NoteGeneration(args))
    test_save_path = os.path.join(args.test_save_path, NoteGeneration(args))

    os.makedirs(logging_path, exist_ok=True)
    # os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=os.path.join(logging_path, 'val.log'),
                        filemode='w')
    logger = logging.getLogger(__name__)

    logging.info(f"Start PIMesh training for {args.epochs} epochs.")

    loss_record = updateLoss(logging_path)
    loss_record.start()

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

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # create Dataset from db file if available else create from folders
    # train_set = InBedPressureDataset(
    #     cfgs,
    #     mode='train'
    # )

    g = torch.Generator()
    g.manual_seed(42)

    cfgs['overlap'] = 0
    val_set = InBedPressureDataset(
        cfgs,
        mode='eval'
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
        num_workers=2
    )

    if args.exp_mode == 'unseen_subject':
        test_loader = None
        test_segments = None
        test_data_len = 0
    else:
        test_set = InBedPressureDataset(
            cfgs,
            mode='test'
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            generator=g,
            num_workers=2
        )
        test_data_len = test_set.get_data_len()
        test_segments = test_set.get_segments()

    # camera
    if not args.use_calibration:
        focal_length = np.sqrt(args.rgb_image_height ** 2 + args.rgb_image_width ** 2)
        focal_length_x, focal_length_y = focal_length, focal_length
    else:
        focal_length_x, focal_length_y = args.focal_length_x, args.focal_length_y

    camera = PerspectiveCamera(focal_length_x=focal_length_x,
                               focal_length_y=focal_length_y,
                               batch_size=args.seqlen,
                               center=torch.Tensor([args.rgb_image_width // 2, args.rgb_image_height // 2]),
                               dtype=dtype).to(device)

    sensor_pitch = args.sensor_pitch
    pressure_reso = args.sensor_size


    camera_silh = OrthographicCameras(
        device=device,
        focal_length=torch.Tensor([[-1 / sensor_pitch[1], 1 / sensor_pitch[0]]]),
        image_size=torch.Tensor([pressure_reso]),
        principal_point=torch.Tensor([[0, pressure_reso[0]]]),
        in_ndc=False)

    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=pressure_reso,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=50,
        bin_size=0,
    )

    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=camera_silh, raster_settings=raster_settings_soft),
        shader=SoftSilhouetteShader())

    # live_loss = updateLoss()

    if args.encoder == 'mae':
        MAEConfigs = dict(
            img_size=args.pi_img_size,
            patch_size=args.mae_patch_size,
            embed_dim=args.mae_encoder_dim,
            depth=args.mae_encoder_depth,
            num_heads=args.mae_encoder_head,
            input_channels=args.mae_input_channel,
            output_channels=args.mae_output_channel,
        )
    else:
        MAEConfigs = None

    if args.temp_encoder == 'gru':
        TEncoderConfigs = dict(
            input_size=args.mae_encoder_dim,
            hidden_size=args.gru_hidden_layers,
            bidirectional=False,
            num_layers=args.gru_layers
        )
    elif args.temp_encoder == 'trans':
        TEncoderConfigs = dict(
            input_feature_len=args.mae_encoder_dim,
            heads=8,
            mlp_hidden_dim=512,
            depth=args.trans_depth,
            drop_path_rate=0.2,
            drop_rate=0.1,
            seqlen=args.seqlen
        )
    elif args.temp_encoder == '1dconv':
        TEncoderConfigs = dict(
            input_feature_len=args.mae_encoder_dim,
            kernel_size=3,
        )
    elif args.temp_encoder == 'fc':
        TEncoderConfigs = None
    elif args.temp_encoder == 'rnn':
        TEncoderConfigs = dict(
            input_feature_len=args.mae_encoder_dim,
            output_feature_len=args.mae_encoder_dim,
            hidden_size=args.gru_hidden_layers,
            num_layers=args.gru_layers
        )

    # model

    model = PIMeshNet(
        seqlen=args.seqlen,
        camera=camera,
        bed_depth=args.bed_depth,
        MAEConfigs=MAEConfigs,
        TEncoderConfigs=TEncoderConfigs,
        feature_len=args.mae_encoder_dim,
        tem_feature_len=args.feature_len,
        encoder_model=args.encoder,
        tem_encoder_model=args.temp_encoder,
        mae_load_pretrain=args.mae_load_pretrain,
        mae_pretrain_ck_path=args.mae_pretrain_ck_path
    )

    ckpt = torch.load(args.test_best_checkpoints, map_location="cpu")['state_dict']
    model.load_state_dict(ckpt)

    body_model = smplx.create('data/models', model_type='smpl').to(device)

    # regressore loss
    loss = HPSPILoss(
        camera=camera,
        camera_silh=camera_silh,
        renderer_silhouette=renderer_silhouette,
        faces=torch.tensor(body_model.faces.astype(np.int64), dtype=torch.long,
                                        device=device).unsqueeze_(0).repeat([args.seqlen * args.batch_size, 1, 1]),
        bed_depth=args.bed_depth,
        sensor_pitch=torch.Tensor([[1 / sensor_pitch[1], 1 / sensor_pitch[0]]]),
        e_pressure_weight=args.e_pressure_weight,
        e_3d_loss_weight=args.e_3d_loss_weight,
        e_2d_loss_weight=args.e_2d_loss_weight,
        e_pose_loss_weight=args.e_pose_loss_weight,
        e_shape_loss_weight=args.e_shape_loss_weight,
        e_trans_loss_weight=args.e_trans_loss_weight,
        e_smooth_pose_weight=args.e_smooth_pose_weight,
        e_smooth_betas_weight=args.e_smooth_betas_weight,
        e_smooth_trans_weight=args.e_smooth_trans_weight,
        e_smooth_joints_weight=args.e_smooth_joints_weight
    )

    # data

    # optimizer and schedule
    num_layers = model.get_num_layers()
    print(num_layers)
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i)
                                                for i in range(num_layers + 2)))
    else:
        assigner = None

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    print(optimizer)

    # ===================== Start Training ===================
    Trainer(
        args=args,
        train_loader=None,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=loss,
        loss_record=loss_record,
        writer=logging,
        checkpoints_path=checkpoints_path,
        exp_mode=args.exp_mode,
        curr_fold=args.curr_fold,
        test_loader=test_loader,
        len_val_set=val_set.get_data_len(),
        len_test_set=test_data_len,
        val_segments=val_set.get_segments(),
        test_segments=test_segments,
        device=device,
        test_save_path=test_save_path
    ).test()

    loss_record.end()

if __name__ == '__main__':

    args = parser_train_config()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if args.encoder == 'mae' and args.exp_mode == 'unseen_subject' and args.curr_fold == 1:
        args.mae_pretrain_ck_path = \
            r'/workspace/wzy1999/MAE4PI/checkpoints/64_64_wzy_unseen_subject_fold_1/vit-mae_losses_0.002430939038340247.pth'
    elif args.encoder == 'mae' and args.exp_mode == 'unseen_subject' and args.curr_fold == 1:
        args.mae_pretrain_ck_path = \
            r'/workspace/wzy1999/MAE4PI/checkpoints/64_64_wzy_unseen_subject_fold_2/vit-mae_losses_0.002543593077318259.pth'
    elif args.encoder == 'mae' and args.exp_mode == 'unseen_subject' and args.curr_fold == 1:
        args.mae_pretrain_ck_path = \
            r'/workspace/wzy1999/MAE4PI/checkpoints/64_64_wzy_unseen_subject_fold_3/vit-mae_losses_0.0025655694757032053.pth'
    elif args.encoder == 'mae' and args.exp_mode == 'unseen_group':
        args.mae_pretrain_ck_path = \
            r'/workspace/wzy1999/MAE4PI/checkpoints/64_64_wzy_unseen_group/vit-mae_losses_0.002624695446788156.pth'

    args.note = 'test'

    # path = r'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/checkpoints/fix_bug/ab_trans/trans_nums'
    # for temp_encoder in ['rnn', 'gru', 'trans']:
    #     model_kp_list = os.listdir(os.path.join(path, temp_encoder))
    #     for model_kp in model_kp_list:
    #         params_list = model_kp.split('_')
    #         if 'unseen_group' in model_kp:
    #             args.exp_mode = 'unseen_group'
    #             args.encoder = params_list[3]
    #             args.temp_encoder = params_list[4]
    #             args.trans_depth = params_list[5]
    #             args.seqlen = int(params_list[9])
    #             args.batch_size = int(params_list[8])
    #             args.cosine = 1
    #             kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
    #             kp_list.sort(reverse=True)
    #             args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
    #             print(args.test_best_checkpoints)
    #             main(args)
    #         else:
    #             args.exp_mode = 'unseen_subject'
    #             args.encoder = params_list[2]
    #             args.temp_encoder = params_list[3]
    #             args.trans_depth = params_list[4]
    #             args.seqlen = int(params_list[8])
    #             args.batch_size = int(params_list[7])
    #             args.cosine = 1
    #             args.curr_fold = int(params_list[1])
    #             kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
    #             kp_list.sort(reverse=True)
    #             args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
    #             print(args.test_best_checkpoints)
    #             main(args)

    # path = r'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/checkpoints/fix_bug/ab_trans/trans_nums'
    # temp_encoder = 'trans'
    # model_kp_list = os.listdir(path)
    # for model_kp in model_kp_list:
    #     params_list = model_kp.split('_')
    #     if 'unseen_group' in model_kp:
    #         args.exp_mode = 'unseen_group'
    #         args.encoder = params_list[3]
    #         args.temp_encoder = params_list[4]
    #         args.trans_depth = int(params_list[5])
    #         args.seqlen = int(params_list[9])
    #         args.batch_size = int(params_list[8])
    #         args.cosine = 1
    #         kp_list = os.listdir(os.path.join(path, model_kp))
    #         kp_list.sort(reverse=True)
    #         args.test_best_checkpoints = os.path.join(path, model_kp, kp_list[0])
    #         print(args.test_best_checkpoints)
    #         main(args)
    #     else:
    #         args.exp_mode = 'unseen_subject'
    #         args.encoder = params_list[2]
    #         args.temp_encoder = params_list[3]
    #         args.trans_depth = int(params_list[4])
    #         args.seqlen = int(params_list[8])
    #         args.batch_size = int(params_list[7])
    #         args.cosine = 1
    #         args.curr_fold = int(params_list[1])
    #         kp_list = os.listdir(os.path.join(path, model_kp))
    #         kp_list.sort(reverse=True)
    #         args.test_best_checkpoints = os.path.join(path, model_kp, kp_list[0])
    #         print(args.test_best_checkpoints)
    #         main(args)

    # path = r'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/checkpoints/fix_bug/mae'
    # for temp_encoder in ['trans']:
    #     model_kp_list = os.listdir(os.path.join(path, temp_encoder))
    #     for model_kp in model_kp_list:
    #         params_list = model_kp.split('_')
    #         if 'unseen_group' in model_kp:
    #             args.exp_mode = 'unseen_group'
    #             args.encoder = params_list[3]
    #             args.temp_encoder = params_list[5]
    #             args.seqlen = int(params_list[9])
    #             args.batch_size = int(params_list[8])
    #             args.mae_not_freeze = True
    #             args.cosine = 1
    #             if params_list[4] == 'True':
    #                 args.mae_load_pretrain = True
    #             elif params_list[4] == 'False':
    #                 args.mae_load_pretrain = False
    #             else:
    #                 args.mae_load_pretrain = True
    #                 args.mae_not_freeze = False
    #             kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
    #             kp_list.sort(reverse=True)
    #             args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
    #             print(args.test_best_checkpoints)
    #             main(args)
    #         else:
    #             args.exp_mode = 'unseen_subject'
    #             args.encoder = params_list[2]
    #             args.temp_encoder = params_list[4]
    #             args.seqlen = int(params_list[8])
    #             args.batch_size = int(params_list[7])
    #             args.cosine = 1
    #             args.curr_fold = int(params_list[1])
    #             args.mae_not_freeze = True
    #             if params_list[3] == 'True':
    #                 args.mae_load_pretrain = True
    #             elif params_list[3] == 'False':
    #                 args.mae_load_pretrain = False
    #             else:
    #                 args.mae_load_pretrain = True
    #                 args.mae_not_freeze = False
    #             kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
    #             kp_list.sort(reverse=True)
    #             args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
    #             print(args.test_best_checkpoints)
    #             main(args)

    # time_count
    path = r'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/checkpoints/fix_bug/resnet'
    for temp_encoder in ['trans']:
        model_kp_list = os.listdir(os.path.join(path, temp_encoder))
        for model_kp in model_kp_list:
            params_list = model_kp.split('_')
            if 'unseen_group' in model_kp and 'resnet50' in model_kp and int(params_list[8]) == 16:
                args.exp_mode = 'unseen_group'
                args.encoder = params_list[3]
                args.temp_encoder = params_list[4]
                args.seqlen = int(params_list[8])
                args.batch_size = 1
                args.cosine = 1
                kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
                kp_list.sort(reverse=True)
                args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
                print(args.test_best_checkpoints)
                main(args)
    # path = r'/workspace/wzy1999/3d_pose_estimation/in_bed_3d_human_estimation/checkpoints/fix_bug/mae'
    # for temp_encoder in ['trans']:
    #     model_kp_list = os.listdir(os.path.join(path, temp_encoder))
    #     for model_kp in model_kp_list:
    #         params_list = model_kp.split('_')
    #         if 'unseen_group' in model_kp  and int(params_list[9]) == 16 and params_list[4] == 'True':
    #             args.exp_mode = 'unseen_group'
    #             args.encoder = params_list[3]
    #             args.temp_encoder = params_list[5]
    #             args.seqlen = int(params_list[9])
    #             args.batch_size = 1
    #             args.mae_not_freeze = True
    #             args.cosine = 1
    #             if params_list[4] == 'True':
    #                 args.mae_load_pretrain = True
    #             elif params_list[4] == 'False':
    #                 args.mae_load_pretrain = False
    #             else:
    #                 args.mae_load_pretrain = True
    #                 args.mae_not_freeze = False
    #             kp_list = os.listdir(os.path.join(path, temp_encoder, model_kp))
    #             kp_list.sort(reverse=True)
    #             args.test_best_checkpoints = os.path.join(path, temp_encoder, model_kp, kp_list[0])
    #             print(args.test_best_checkpoints)
    #             main(args)