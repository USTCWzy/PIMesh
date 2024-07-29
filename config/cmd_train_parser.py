
import argparse

def parser_train_config():

    parser = argparse.ArgumentParser()

    # dataset & experiment
    parser.add_argument('--dataset_path',
                        # default='F:\dataset\TIP\wzy_opt_dataset',
                        default=r'/workspace/wzy1999/Public_Dataset/wzy_opt_dataset',
                        type=str,
                        help='dataset path')
    parser.add_argument('--curr_fold',
                        default=1,
                        type=int,
                        help='curr fold for 3-fold validation')
    parser.add_argument('--exp_mode',
                        default='unseen_subject',
                        type=str,
                        help='unseen_subject or unseen_group')
    parser.add_argument('--gpu',
                        default=1,
                        type=int,
                        help='curr gpu')
    parser.add_argument('--encoder',
                        default='mae',
                        type=str,
                        help='which encoder')
    parser.add_argument('--temp_encoder',
                        default='gru',
                        type=str,
                        help='which sequence model')

    # weights
    parser.add_argument('--e_pressure_weight', default=0., type=float, help='pressure loss weight')
    parser.add_argument('--e_3d_loss_weight', default=10., type=float, help='3d joint loss weight')
    parser.add_argument('--e_2d_loss_weight', default=0, type=float, help='2d projection loss weight')
    parser.add_argument('--e_pose_loss_weight', default=100., type=float, help='pose loss weight')
    parser.add_argument('--e_shape_loss_weight', default=100, type=float, help='shape loss weight')
    parser.add_argument('--e_trans_loss_weight', default=300., type=float, help='trans loss weight')
    parser.add_argument('--e_smooth_pose_weight', default=50., type=float, help='smooth pose loss weight')
    parser.add_argument('--e_smooth_betas_weight', default=50., type=float, help='smooth betas loss weight')
    parser.add_argument('--e_smooth_trans_weight', default=50., type=float, help='smooth trans loss weight')
    parser.add_argument('--e_smooth_joints_weight', default=20., type=float, help='smooth joints loss weight')

    # checkpoints and logging
    parser.add_argument('--logging_path',
                        default='log',
                        type=str,
                        help='dataset path')
    parser.add_argument('--checkpoints_path',
                        default='checkpoints',
                        type=str,
                        help='dataset path')
    parser.add_argument('--test_save_path',
                        default='trans_nums',
                        type=str,
                        help='test save path')
    parser.add_argument('--test_best_checkpoints',
                        type=str,
                        help='best checkpoints')

    # sensor
    parser.add_argument('--sensor_size',
                        default=[56, 40],
                        type=list,
                        help='sensor array size')
    parser.add_argument('--sensor_pitch',
                        default=[0.0311, 0.0195],
                        type=list,
                        help='sensor array pitch')


    #opt
    parser.add_argument('--lr',
                        default=5e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        default=5e-3,
                        type=float,
                        help='weight decay')
    parser.add_argument('--cosine',
                        default=True,
                        type=bool,
                        help='cosine lr')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='epochs')
    parser.add_argument('--warmup_epochs',
                        default=5,
                        type=int,
                        help='warm_up epochs')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--val_batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_smplify_iters',
                        default=[200, 290],
                        type=list,
                        help='number of iters')
    parser.add_argument('--seqlen',
                        default=16,
                        type=int,
                        help='sequence length')
    parser.add_argument('--overlap',
                        default=0.25,
                        type=float,
                        help='overlap ratio')
    parser.add_argument('--opt',
                        default='AdamW',
                        type=str,
                        help='optimizer')

    parser.add_argument('--note', default='', type=str)

    # network
    parser.add_argument('--pi_img_size',
                        default=(64, 64),
                        help='image size')

    # mae
    # parser.add_argument('--pi_img_size', default=(64, 64), help='image size')
    parser.add_argument('--mae_patch_size', default=8, type=int, help='mae patch size')
    parser.add_argument('--mae_encoder_dim', default=768, type=int, help='mae encoder dim')
    parser.add_argument('--mae_encoder_depth', default=12, type=int, help='vit depth')
    parser.add_argument('--mae_encoder_head', default=12, type=int, help='no. of attention heads')
    parser.add_argument('--mae_input_channel', default=1, type=int, help='input channel')
    parser.add_argument('--mae_output_channel', default=1, type=int, help='output channel')
    parser.add_argument('--layer_decay', default=0.75, type=float, help='vit layer decay')
    parser.add_argument('--mae_load_pretrain', default=True, type=str, help='whether to load pretrained params')
    parser.add_argument('--mae_not_freeze', default=True, type=str, help='whether to freeze encoder')
    parser.add_argument('--mae_pretrain_ck_path', default=None, type=str, help='pretrained params path')

    #gru
    # parser.add_argument('--pi_img_size', default=(64, 64), help='image size')
    parser.add_argument('--gru_hidden_layers', default=768, type=int, help='gru hidden layer length')
    parser.add_argument('--gru_layers', default=1, type=int, help='no. of gru layers')

    parser.add_argument('--trans_depth', default=2, type=int, help='no. of trans layers')

    # regressor
    parser.add_argument('--feature_len', default=768, type=int, help='regressor feature length')

    # weight
    parser.add_argument('--contact_loss_weight',
                        default=2000,
                        type=float,
                        help='contact loss weight')

    # contact
    parser.add_argument('--geo_thres',
                        default=0.3,
                        type=float,
                        help='geodis threshold')
    parser.add_argument('--eucl_thres',
                        default=0.02,
                        type=float,
                        help='eucl distance threshold')
    parser.add_argument('--geodis_smpl_path',
                        default='data/essentials/geodesics/smpl/smpl_neutral_geodesic_dist.npy',
                        type=str,
                        help='initial geodis file path')
    parser.add_argument('--essentials_dir',
                        default='data/essentials',
                        type=str,
                        help='path to store smpl-related files')

    # smpl
    parser.add_argument('--smpl_model',
                        default='smpl',
                        type=str,
                        help='smpl/smplx/smplh')
    parser.add_argument('--smpl_gender',
                        default='neutral',
                        type=str,
                        help='male/female/neutral')

    # camera
    parser.add_argument('--rgb_image_height',
                        default=1920,
                        type=float,
                        help='image height.')
    parser.add_argument('--rgb_image_width',
                        default=1080,
                        type=float,
                        help='image width.')
    parser.add_argument('--focal_length_x',
                        default=941,
                        type=float,
                        help='Value of focal length x-axis.')
    parser.add_argument('--focal_length_y',
                        default=941,
                        type=float,
                        help='Value of focal length y-axis.')
    parser.add_argument('--use_calibration',
                        default=True,
                        type=bool,
                        help='whether to use calibrated focal length')
    parser.add_argument('--bed_depth',
                        default=1.66,
                        type=float,
                        help='distance from the bed to camera')

    args = parser.parse_args()

    return args
