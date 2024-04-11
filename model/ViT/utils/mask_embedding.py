import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast as autocast

from torchvision.transforms import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def ShuffleIndex(index: list, sample_ratio: float):
    sample_list = []
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    else:
        remain_length = int((1 - sample_ratio) * len(index))
        temp_index = index.copy()
        while len(temp_index) > remain_length:
            sample = random.choice(temp_index)
            sample_list.append(sample)
            temp_index.remove(sample)

        mask_list = [x for x in index if
                     x not in sample_list]  # get to remain index not in cls token and not in sample_index
        assert len(sample_list) == int(len(index) * sample_ratio), "sample length must be same as the ratio!!!"
    return sample_list, mask_list


def MaskEmbedding(token_emb, mask_ratio):
    """get the mask embedding after patch_emb + pos_emb
    """
    token_length = token_emb.shape[1]
    token_index = [x for x in range(1, token_length)]
    # print(len(token_index))
    mask_index, sample_index = ShuffleIndex(token_index, mask_ratio)
    token_sample = [0] + sample_index

    x = token_emb[:, token_sample, :]
    return x, sample_index, mask_index


class UnMaskEmbedding(nn.Module):
    """get the mask embedding from the image -> 127 to embedding, before the position embedding
    """

    def __init__(self, input_size, embed_dim, in_chans, patch_size, num_patches):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.num_patches = num_patches
        # used for mask images
        if isinstance(input_size, tuple):
            self.raw_inputs = torch.ones((in_chans, input_size[0], input_size[1])) * 127. / 255
        elif isinstance(input_size, int):
            self.raw_inputs = torch.ones((in_chans, input_size, input_size)) * 127. / 255
        self.raw_inputs = transforms.Normalize(mean=0.3286797, std=0.3286797)(self.raw_inputs)
        self.raw_inputs = self.raw_inputs.unsqueeze(0)
        self.raw_inputs = self.raw_inputs.cuda()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, sample_index, mask_index):
        b, _, _ = x.shape
        self.raw_inputs = self.raw_inputs.expand(b, -1, -1, -1)
        decoder_embedding = nn.Parameter(torch.zeros((b, 1 + self.num_patches, self.embed_dim))).cuda()
        with autocast():
            embedding = self.proj(self.raw_inputs)  # b, c, h, w

            b, c, h, w = embedding.shape
            patch_embedding = embedding.view(b, -1, c)[0, 0, :]

            # include the cls token
            # print(x.dtype)
            # print(decoder_embedding.dtype)
            if x.dtype == torch.float16:
                decoder_embedding = decoder_embedding.half()

            decoder_embedding[:, [0] + sample_index, :] = x
            decoder_embedding[:, mask_index, :] = patch_embedding

        return decoder_embedding


if __name__ == '__main__':
    a = [x for x in range(196)]
    sample = ShuffleIndex(a, 0.75)
    print(sample)


