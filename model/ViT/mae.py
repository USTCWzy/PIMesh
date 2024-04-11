"""
Masked Autoencoders Are Scalable Vision Learners
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.patch_embed import PatchEmbed, PositionEmbed
from .vit import VisionTransformer
from einops import rearrange

class MaskTransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        '''
        construct the normalization for each patches
        '''
        super(MaskTransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class MAEVisionTransformers(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 encoder_dim = 1024,
                 encoder_depth = 24,
                 encoder_heads = 16,
                 decoder_dim =512,
                 decoder_depth = 8,
                 decoder_heads = 16,
                 mask_ratio = 0.75,
                 flag = 0,
                 input_channel=3,
                 output_channel=3
                 ):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(img_size, tuple):
            self.num_patch = (img_size[0] // self.patch_size, img_size[1] // self.patch_size)
        else:
            self.num_patch = (img_size // self.patch_size, img_size // self.patch_size)
        self.flag = flag
        base_cfg = dict(
            img_size=img_size,
            in_chans=input_channel,
            num_classes=1000,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            embed_layer=PatchEmbed,
            pos_embed='cosine',
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pool='mean',
        )
        encoder_model_dict = dict(
            patch_size=self.patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            classification=False,
            vit_type='encoder',
            mask_ratio=mask_ratio,
        )
        decoder_model_dict = dict(
            patch_size=self.patch_size,
            embed_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            classification=False,
            vit_type='decoder',
            mask_ratio=mask_ratio,
        )

        ENCODER_MODEL_CFG = {**base_cfg, **encoder_model_dict}
        DECODER_MODEL_CFG = {**base_cfg, **decoder_model_dict}

        self.Encoder = VisionTransformer(**ENCODER_MODEL_CFG)
        self.Decoder = VisionTransformer(**DECODER_MODEL_CFG)

        output_dim = patch_size * patch_size * output_channel

        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.restruction = nn.Linear(decoder_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.patch_norm = MaskTransLayerNorm(output_dim)

        #restore image from unconv
        self.unconv = nn.ConvTranspose2d(output_dim, output_channel, patch_size, patch_size)
        self.apply(self.init_weights)
        # import pdb;pdb.set_trace()

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

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        #  batch, c, h, w
        #import pdb;pdb.set_trace()
        norm_embedding, sample_index, mask_index = self.Encoder.autoencoder(x)
        proj_embedding = self.proj(norm_embedding)
        decode_embedding = self.Decoder.decoder(proj_embedding, sample_index, mask_index)
        outputs = self.restruction(decode_embedding)

        cls_token = outputs[:, 0, :]
        image_token = outputs[:, 1:, :]
        image_norm_token = self.patch_norm(image_token)
        n, l, dim = image_norm_token.shape
        image_norm_token = image_norm_token.view(-1, self.num_patch[0], self.num_patch[1], dim).permute(0, 3, 1, 2)
        restore_image = self.unconv(image_norm_token)
        return restore_image, mask_index

class VisionTransformers(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=192,
                 depth=12,
                 num_heads=3,
                 num_classes=1000,
                 input_channels=1,
                 output_channels=1,
                 pretrain_path=None
                 ):
        super(VisionTransformers, self).__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.pretrain_path = pretrain_path
        base_cfg = dict(
            img_size=self.img_size,
            in_chans=input_channels,
            num_classes=self.num_classes,
            classification=False,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            embed_layer=PatchEmbed,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            depth=self.depth,
            patch_size=self.patch_size,
            pos_embed="cosine",
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pool='cls',
        )

        self.model = VisionTransformer(**base_cfg)
        self.model.apply(self.init_weights)
        if pretrain_path is not None:
            self._load_mae_pretrain()

    def forward(self, x):
        return self.model(x)

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

    def _load_mae_pretrain(self, mae_model_path=None):
        state_dict = torch.load(mae_model_path,
                                map_location="cpu")['state_dict']
        # state_dict = torch.load("/workspace/wzy1999/MAE4PI/checkpoints/weights/pretrain.pth",
        #                         map_location="cpu")['state_dict']
        ckpt_state_dict = {}
        for key, value in state_dict.items():
            if 'Encoder.' in key:
                if key[8:] in self.model.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value

        for key, value in self.model.state_dict().items():
            if key not in ckpt_state_dict.keys():
                print('There only the FC have no load pretrain!!!', key)

        state = self.model.state_dict()
        state.update(ckpt_state_dict)
        self.model.load_state_dict(state)
        print("model load the mae pretrain!!!")

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.model.blocks)

if __name__ == '__main__':
    pass
