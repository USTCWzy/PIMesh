import os
import torch
import os.path as osp
import torch.nn as nn

from torchvision import models
from functools import partial
import torch.nn.functional as F
from .ViT.mae import VisionTransformers as FeatureExtractor
from .spin import Regressor
from .transBlock import Block
from .base_encoder import *
from .temp_encoder import *

class PIMeshNet(nn.Module):

    def __init__(
            self,
            seqlen,
            camera,
            bed_depth,
            TEncoderConfigs,
            MAEConfigs=None,
            feature_len=1024,
            tem_feature_len=1024,
            encoder_model='mae',
            tem_encoder_model='gru',
            mae_load_pretrain=True,
            mae_pretrain_ck_path=''
    ):

        super(PIMeshNet, self).__init__()

        self.seqlen = seqlen
        self.encoder_model = encoder_model
        self.feature_len = feature_len

        if self.encoder_model == 'mae':
            self.encoder = MAEEncoder(MAEConfigs)
        elif self.encoder_model[:6] == 'resnet':
            self.encoder = RESNETEncoder(encoder_model, feature_len)

        if tem_encoder_model == 'gru':
            self.temp_encoder = TemporalGRUEncoder(
                TEncoderConfigs,
                add_linear=True,
                use_residual=True,
                feature_len=feature_len,
                model_type=tem_encoder_model
            )
        elif tem_encoder_model == 'fc':
            self.temp_encoder = TemporalFCEncoder(
                seqlen=self.seqlen,
                feature_len=self.feature_len
            )
        elif tem_encoder_model == 'trans':
            self.temp_encoder = TemporalTransEncoder(
                TEncoderConfigs
            )
        elif tem_encoder_model == '1dconv':
            self.temp_encoder = Temporal1DConvEncoder(
                TEncoderConfigs
            )
        elif tem_encoder_model == 'rnn':
            self.temp_encoder = TemporalRNNEncoder(
                TEncoderConfigs,
                add_linear=True,
                use_residual=True
            )
        self.regressor = Regressor(camera, bed_depth, feature_len)

        self.apply(self.init_weights)

        if self.encoder_model == 'mae' and mae_load_pretrain:
            self.encoder.model._load_mae_pretrain(mae_pretrain_ck_path)

    def forward(self, x, is_train=True):
        # import pdb;pdb.set_trace()
        batch_size, seqlen, h, w = x.shape

        x = self.encoder(x)

        x = self.temp_encoder(x)

        x = x.reshape(-1, self.feature_len)

        smpl_output = self.regressor(x)

        if is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
                s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
        return smpl_output

    def get_num_layers(self):
        if self.encoder_model == 'mae':
            return self.encoder.model.get_num_layers()
        else:
            return 0
    def no_weight_decay(self):
        if self.encoder_model == 'mae':
            return self.encoder.model.no_weight_decay()
        else:
            return []

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


