import torch
import torch.nn as nn

from functools import partial
import torch.nn.functional as F
from .transBlock import Block

class TemporalGRUEncoder(nn.Module):

    def __init__(self, model_params, add_linear, use_residual, feature_len=1024, model_type='gru'):
        super(TemporalGRUEncoder, self).__init__()

        if model_type == 'gru':
            self.temp_encoder = nn.GRU(
                **model_params
            )

        if add_linear:
            self.linear = nn.Linear(model_params['hidden_size'], feature_len)
        self.feature_len = feature_len
        self.use_residual = use_residual

    def forward(self, x):

        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.temp_encoder(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == self.feature_len:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y

class TemporalFCEncoder(nn.Module):
    def __init__(self, seqlen, feature_len):
        super(TemporalFCEncoder, self).__init__()

        self.seqlen = seqlen
        self.feature_len = feature_len

        self.TempEncoder = nn.Sequential(
            nn.Linear(seqlen * feature_len, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, seqlen * feature_len)
        )

    def forward(self, x):
        n, t, f = x.shape
        x = self.TempEncoder(x.reshape(n, t * f)).reshape(n, t, f)
        return x

class TemporalTransEncoder(nn.Module):

    def __init__(self, model_params):
        super(TemporalTransEncoder, self).__init__()

        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, model_params['seqlen'], model_params['input_feature_len']))

        dpr = [x.item() for x in torch.linspace(0, model_params['drop_path_rate'], model_params['depth'])]

        self.blocks = nn.ModuleList([
            Block(
                dim=model_params['input_feature_len'], num_heads=model_params['heads'], mlp_hidden_dim=model_params['mlp_hidden_dim'], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=model_params['drop_rate'], attn_drop=model_params['drop_path_rate'], drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(model_params['depth'])])

        self.norm = norm_layer(model_params['input_feature_len'])

    def forward(self, x):
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class Temporal1DConvEncoder(nn.Module):
    def __init__(self, model_params):
        super(Temporal1DConvEncoder, self).__init__()

        self.TempEncoder = nn.Sequential(
            nn.Conv1d(
                in_channels=model_params['input_feature_len'],
                out_channels=model_params['input_feature_len'],
                kernel_size=model_params['kernel_size'],
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.TempEncoder(x)
        return x.permute(0, 2, 1)

class TemporalRNNEncoder(nn.Module):
    def __init__(self, model_params, add_linear, use_residual):
        super(TemporalRNNEncoder, self).__init__()

        self.temp_encoder = nn.RNN(
            input_size=model_params['input_feature_len'],
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers'],
            batch_first=True
        )

        if add_linear:
            self.linear = nn.Linear(model_params['hidden_size'], model_params['output_feature_len'])
            self.feature_len = model_params['output_feature_len']
        else:
            self.feature_len = model_params['hidden_size']
        self.use_residual = use_residual

    def forward(self, x):

        # import pdb;pdb.set_trace()
        n, t, f = x.shape
        y, _ = self.temp_encoder(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.reshape(-1, y.size(-1)))
            y = y.view(n, t, f)
        if self.use_residual and y.shape[-1] == self.feature_len:
            y = y + x
        return y
