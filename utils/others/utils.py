import os
import math
import torch
import random
import numpy as np
def NoteGeneration(args):

    note = 'PIMesh'

    exp = '_' + args.exp_mode
    if args.exp_mode == 'unseen_subject':
        exp = '_' + str(args.curr_fold)

    encoder = '_' + args.encoder
    if args.encoder == 'mae':
        if not args.mae_load_pretrain:
            encoder += '_' + str(args.mae_load_pretrain)
        elif args.mae_not_freeze:
            encoder += '_' + str(args.mae_load_pretrain)
        else:
            encoder += '_freeze'


    temp_encoder = '_' + args.temp_encoder + '_' + str(args.trans_depth)

    reg = '_spin'

    exp_params = '_' + str(args.lr) + '_' + str(args.batch_size) + '_' + str(args.seqlen) + '_' + str(args.overlap)

    cosine = '_'
    if args.cosine:
        cosine += 'cosine'
    else:
        cosine += 'linear'

    notes = ''
    if len(args.note) > 0:
        notes = '_' + args.note


    weight = '_' + str(args.e_3d_loss_weight) + '_' + str(args.e_2d_loss_weight) + '_' + str(args.e_pose_loss_weight) + \
             '_' + str(args.e_shape_loss_weight) + '_' + str(args.e_trans_loss_weight) + '_' + str(args.e_smooth_pose_weight) + \
             '_' + str(args.e_smooth_betas_weight) + '_' + str(args.e_smooth_trans_weight) + '_' + str(args.e_smooth_joints_weight) + \
             '_' + str(args.e_pressure_weight)

    return note + exp + encoder + temp_encoder + reg + exp_params + weight + cosine + notes

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True



def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = args.epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
    elif epoch < int(0.4 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(0.9 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj

def cosine_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    total_epochs = args.epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1/2 * (1 + math.cos(batch_iter * math.pi /
                                     ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj

    return args.lr * lr_adj

def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7: ]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

