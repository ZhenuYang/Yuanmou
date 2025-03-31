import torch
import torch.nn as nn
import numpy as np
import random
from utils import ps_inv
from logger import logger
from collections import defaultdict
from itertools import combinations

from utils import get_stitch_configs, ps_inv


class StitchingLayer(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super().__init__()

        if in_dim == out_dim:
            self.transform = nn.Identity()
        elif in_dim != out_dim:
            self.transform = nn.Linear(in_dim, out_dim)

    def init_stitch_weights_bias(self, weight, bias=None):
        self.transform.weight.data.copy_(weight)
        if bias != None:
            self.transform.bias.data.copy_(bias)

    def forward(self, x):
        x = self.transform(x)
        return x

class LearngenePool(nn.Module):
    '''
    Stitching from learngene Pool
    '''
    def __init__(self, learngene_instances, nearest_stitching=False, blk_legth=0):
        super(LearngenePool, self).__init__()
        self.blk_length = blk_legth
        self.instances = nn.ModuleList(learngene_instances)
        self.instances_depths = [len(instances.blocks) for instances in self.instances]  # get the number of blocks of each learngene instance

        blk_stitch_cfgs, num_stitches = get_stitch_configs(self.instances_depths[0], kernel_size=2, stride=1,
                                                           num_models=len(self.instances), nearest_stitching=nearest_stitching)
        self.num_stitches = num_stitches

        candidate_combinations = list(combinations(list(range(len(learngene_instances))), 2))  # [(0, 1), (0, 2), (1, 2)]
        if nearest_stitching:
            candidate_combinations.pop(candidate_combinations.index((0, 2)))  # [(0, 1), (0, 2), (1, 2)]
        self.candidate_combinations = candidate_combinations  # [(0, 1), (0, 2), (1, 2)]

        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}
        for i, cand in enumerate(candidate_combinations):
            front, end = cand
            self.stitch_layers.append(
                nn.ModuleList([StitchingLayer(self.instances[front].embed_dim, self.instances[end].embed_dim) for _ in
                               range(num_stitches)])
            )
            self.stitching_map_id[f'{front}-{end}'] = i

        self.stitch_configs = {i: cfg for i, cfg in enumerate(blk_stitch_cfgs)}
        # logger.info(self.stitch_configs)
        self.num_configs = len(blk_stitch_cfgs)
        self.stitch_config_id = 0

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id


    def initialize_stitching_weights(self, x, mode='ours'):
        if mode == 'snnet':
            stitching_dicts = defaultdict(set)
            for id, config in self.stitch_configs.items():
                if len(config['comb_id']) == 1:
                    continue

                # each stitching layer is shared among neighboring blocks, thus it handles different stitching path.
                stitching_dicts[config['stitch_layers'][0]].add(config['stitch_cfgs'][0])  # {}

            for i, combo in enumerate(self.candidate_combinations):
                front, end = combo

                # extract feature maps from the blocks of anchors
                with torch.no_grad():
                    front_features = self.instances[front].extract_block_features(x)
                    end_features = self.instances[end].extract_block_features(x)

                for stitch_layer_id, stitch_positions in stitching_dicts.items():
                    weight_candidates = []
                    bias_candidates = []
                    for front_id, end_id in stitch_positions:
                        front_blk_feat = front_features[front_id]
                        end_blk_feat = end_features[end_id - 1]

                        # solve the least square problem to get the weights and bias
                        w, b = ps_inv(front_blk_feat, end_blk_feat)
                        weight_candidates.append(w)
                        bias_candidates.append(b)

                    # since each stitching layer is shared among different stitching paths, we average the weights and bias
                    weights = torch.stack(weight_candidates).mean(dim=0)
                    bias = torch.stack(bias_candidates).mean(dim=0)

                    self.stitch_layers[i][int(stitch_layer_id)].init_stitch_weights_bias(weights, bias)
                    logger.info(f'Initialized Stitching Model {front} to Model {end}, Layer {stitch_layer_id}')

        elif mode == 'ours':
            ckpt_paths = [
                './train_results/distill/deit_base_patch16_224-deit_tiny6_patch16_224/LearngenePool_[front,end]/TransLayerMap_checkpoint.pth',
                './train_results/distill/deit_base_patch16_224-deit_small6_patch16_224/LearngenePool_[end]/TransLayerMap_checkpoint.pth',]

            tiny_base_ckpt = torch.load(ckpt_paths[0], map_location='cpu')['model']
            total_tiny_base_ckpt_weight = []
            # total_tiny_base_ckpt_bias = []
            for i in range(3):
                total_tiny_base_ckpt_weight.append(tiny_base_ckpt['{}.transform.weight'.format(i)])
                # total_tiny_base_ckpt_bias = torch.cat(tiny_base_ckpt['{}.transform.bias'.format(i)])

            mean_tiny_base_ckpt_weight = (torch.stack(total_tiny_base_ckpt_weight).mean(dim=0)).permute(1, 0)
            # mean_tiny_base_ckpt_bias = torch.stack(total_tiny_base_ckpt_bias)
            for j in range(len(self.stitch_layers[1])):  # initialize stitching layers from 0-2
                self.stitch_layers[1][j].init_stitch_weights_bias(mean_tiny_base_ckpt_weight, None)

            # small_base_ckpt = torch.load(ckpt_paths[1], map_location='cpu')['model']
            # mean_small_base_ckpt_weight = small_base_ckpt['0.transform.weight'].permute(1, 0)
            # # mean_small_base_ckpt_bias = torch.stack(small_base_ckpt['0.transform.bias'], small_base_ckpt['0.transform.bias'])
            # for j in range(len(self.stitch_layers[2])):  # initialize stitching layers from 1-2
            #     self.stitch_layers[2][j].init_stitch_weights_bias(mean_small_base_ckpt_weight, None)
            #
            # mean_tiny_small_ckpt_weight = torch.mm(small_base_ckpt['0.transform.weight'], mean_tiny_base_ckpt_weight)
            # # mean_tiny_small_ckpt_bias = torch.mm(mean_tiny_base_ckpt_bias, small_base_ckpt[1][1])
            # for j in range(len(self.stitch_layers[0])):  # initialize stitching layers from 0-1
            #     self.stitch_layers[0][j].init_stitch_weights_bias(mean_tiny_small_ckpt_weight, None)

    def forward(self, x):
        if self.training:
            assert self.blk_length == 6 or self.blk_length == 9
            if self.blk_length == 6:
                stitch_cfg_ids = [0, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # blk_length=6
            elif self.blk_length == 9:
                stitch_cfg_ids = [0, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52] # blk_length=9

            stitch_cfg_id = random.choice(stitch_cfg_ids)
            # stitch_cfg_id = np.random.randint(0, self.num_configs)  # random sampling during training
        else:
            stitch_cfg_id = self.stitch_config_id
        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']  # (0, 1)
        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']  # [(2, 3)]
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']  # [2]

        # 如果comb_id是1，那么表示不缝合，那就直接返回该网络的输出
        if len(comb_id) == 1:
            # simply forward the instance
            out, _, _, _ = self.instances[comb_id[0]].forward_features(x)
            return out

        # 如果comb_id!=1，表示有两个网络之间的缝合，首先对图片进行patch_embed操作
        x = self.instances[comb_id[0]].forward_patch_embed(x)

        front_id = 0
        for i, cfg in enumerate(stitch_cfgs):
            end_id = cfg[0] + 1

            # 经过第一个网络的前半部分
            for blk in self.instances[comb_id[i]].blocks[front_id:end_id]:
                x = blk(x)

            # Stich layer的部分
            front_id = cfg[1]
            sl_id = stitch_layer_ids[i]
            key = str(comb_id[i]) + '-' + str(comb_id[i + 1])
            stitch_projection_id = self.stitching_map_id[key]
            x = self.stitch_layers[stitch_projection_id][sl_id](x)

        # 经过第二个网络的后半部分
        for blk in self.instances[comb_id[-1]].blocks[front_id:]:
            x = blk(x)

        x = self.instances[comb_id[-1]].forward_head(x)

        return x

    def get_model_size(self, stitch_cfg_id):
        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        if len(comb_id) == 1:
            return sum(p.numel() for p in self.instances[comb_id[0]].parameters())

        total_params = 0
        total_params += sum(p.numel() for p in self.instances[comb_id[0]].patch_embed.parameters())

        front_id = 0

        stitch_params = 0

        for i, cfg in enumerate(stitch_cfgs):
            end_id = cfg[0] + 1
            for blk in self.instances[comb_id[i]].blocks[front_id:end_id]:
                total_params += sum(p.numel() for p in blk.parameters())

            front_id = cfg[1]
            sl_id = stitch_layer_ids[i]
            key = str(comb_id[i]) + '-' + str(comb_id[i + 1])
            stitch_projection_id = self.stitching_map_id[key]
            stitch_params += sum(p.numel() for p in self.stitch_layers[stitch_projection_id][sl_id].parameters())
        total_params = total_params + stitch_params
        logger.info('stitch layer params: {}'.format(stitch_params))

        for blk in self.instances[comb_id[-1]].blocks[front_id:]:
            total_params += sum(p.numel() for p in blk.parameters())

        total_params += sum(p.numel() for p in self.instances[comb_id[-1]].head.parameters())
        total_params += sum(p.numel() for p in self.instances[comb_id[-1]].norm.parameters())
        return total_params

