from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
import pdb
from transformers import BertTokenizer, BertModel
import numpy as np

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # Create a trajectory idx context encoder
        traj_context_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        self.traj_context_encoder = traj_context_encoder
        # context_encoder = nn.Embedding(2, dsed)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.lang_context_encoder = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True)
        
        # Create MLP for encoding the language embedding
        dbert = 768
        lang_context_encoder = nn.Sequential(
            nn.Linear(dbert, dbert * 2),
            nn.Mish(),
            nn.Linear(dbert * 2, dbert),
            nn.Mish(),
            nn.Linear(dbert, dsed)
        )
        self.lang_context_encoder = lang_context_encoder
        
        


        # cond_dim = dsed
        cond_dim = dsed + dsed + dsed
        # cond_dim = dsed + dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.global_cond_dim = global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        # self.context_encoder = context_encoder
        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.cluster_centers = [98,63]
        self.n_clusters = 2
        self.cluster_centers = ['right', 'left']
        # self.n_clusters = 5
        # self.cluster_centers = [34, 11, 68, 25, 93]
        # self.n_clusters = 100
        # self.cluster_centers = [54, 63, 38, 6, 75, 93, 57, 68, 65, 33, 21, 16, 96, 1, 4, 24, 14, 2, 94, 71, 42, 45, 17, 60, 80, 19, 22, 50, 40, 27, 74, 66, 35, 53, 5, 76, 81, 64, 18, 55, 62, 10, 15, 90, 72, 8, 31, 84, 43, 85, 11, 9, 79, 73, 82, 28, 83, 56, 49, 98, 26, 46, 91, 78, 52, 37, 88, 36, 41, 0, 13, 44, 29, 32, 92, 97, 25, 3, 51, 58, 12, 20, 67, 47, 86, 95, 61, 70, 48, 34, 77, 59, 30, 23, 7, 99, 39, 89, 87, 69]

        # self.create_weight_transformer(self.n_clusters, self.cluster_centers)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def create_obs_transformer(self):
        global_cond_dim = self.global_cond_dim
        self.obs_transformer = nn.Sequential(
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim*2),
            nn.Mish(),
            nn.Linear(global_cond_dim*2, global_cond_dim*3),
            nn.Mish(),
            nn.Linear(global_cond_dim*3, global_cond_dim*2),
            nn.Mish(),
            nn.Linear(global_cond_dim*2, global_cond_dim),
        )
        # move to device
        self.obs_transformer = self.obs_transformer.to(next(self.parameters()).device)

    def create_weight_transformer(self, n_clusters, cluster_centers):
        # self.weight_transformer = nn.Sequential(
        #     nn.Linear(size_train, size_train),
        #     nn.Softmax(dim=-1)
        # )
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        
        
        global_cond_dim = self.global_cond_dim
        self.weight_transformer = nn.Sequential(
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim*2),
            nn.Mish(),
            nn.Linear(global_cond_dim*2, global_cond_dim*3),
            nn.Mish(),
            nn.Linear(global_cond_dim*3, global_cond_dim*2),
            nn.Mish(),
            nn.Linear(global_cond_dim*2, global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, n_clusters),
            nn.Softmax(dim=-1)
        )
        # move to device
        self.weight_transformer = self.weight_transformer.to(next(self.parameters()).device)

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, lang_context_cond=None, traj_context_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        context_cond: (B,1)
        output: (B,T,input_dim)
        """
        
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # pdb.set_trace()
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        # Handle trajectory conditioning
        traj_context_cond = None
        if traj_context_cond is not None:
            # pdb.set_trace()
            # type is int or numpy int
            if type(traj_context_cond) == int or type(traj_context_cond) == np.int64:
                conditioning_context = torch.tensor([traj_context_cond], dtype=torch.long, device=sample.device)
            else:
                conditioning_context = traj_context_cond[:,0,0]
            # convert to long
            conditioning_context = conditioning_context.long()
            if not torch.is_tensor(conditioning_context):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                conditioning_context = torch.tensor([conditioning_context], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(conditioning_context) and len(conditioning_context.shape) == 0:
                conditioning_context = conditioning_context[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            conditioning_context = conditioning_context.expand(sample.shape[0])
            traj_global_conditioning_context = self.traj_context_encoder(conditioning_context)
        else:
            traj_global_conditioning_context = torch.zeros_like(global_feature)

        # Handle language conditioning
        if lang_context_cond is not None:
            if len(lang_context_cond.shape) == 2:
                lang_global_conditioning_context = lang_context_cond
            else:
                lang_global_conditioning_context = lang_context_cond[:,0,:]
            # pdb.set_trace()
            # convert to float
            lang_global_conditioning_context = lang_global_conditioning_context.float()
            # print("lang_global_conditioning_context", lang_global_conditioning_context.shape)
            lang_global_conditioning_context = self.lang_context_encoder(lang_global_conditioning_context)
        else:
            lang_global_conditioning_context = torch.zeros_like(global_feature)

        # pdb.set_trace()
        # global_cond = self.obs_transformer(global_cond)
        # lang_global_conditioning_context = None
        
        if global_cond is not None and lang_global_conditioning_context is not None and traj_global_conditioning_context is not None:
            # pdb.set_trace()
            global_feature = torch.cat([
                global_feature, lang_global_conditioning_context, traj_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None and lang_global_conditioning_context is not None:
            global_feature = torch.cat([
                global_feature, lang_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None and traj_global_conditioning_context is not None:
            global_feature = torch.cat([
                global_feature, traj_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

    def get_obs_encoding(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, lang_context_cond=None, traj_context_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        context_cond: (B,1)
        output: (B,T,input_dim)
        """
        
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # pdb.set_trace()
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        # Handle trajectory conditioning
        if traj_context_cond is not None:
            conditioning_context = traj_context_cond[:,0,0]
            # convert to long
            conditioning_context = conditioning_context.long()
            if not torch.is_tensor(conditioning_context):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                conditioning_context = torch.tensor([conditioning_context], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(conditioning_context) and len(conditioning_context.shape) == 0:
                conditioning_context = conditioning_context[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            conditioning_context = conditioning_context.expand(sample.shape[0])
            traj_global_conditioning_context = self.traj_context_encoder(conditioning_context)
        else:
            traj_global_conditioning_context = torch.zeros_like(global_feature)

        # Handle language conditioning
        if lang_context_cond is not None:
            lang_global_conditioning_context = lang_context_cond[:,0,:]
            # pdb.set_trace()
            # convert to float
            lang_global_conditioning_context = lang_global_conditioning_context.float()
            # print("lang_global_conditioning_context", lang_global_conditioning_context.shape)
            lang_global_conditioning_context = self.lang_context_encoder(lang_global_conditioning_context)
        else:
            lang_global_conditioning_context = torch.zeros_like(global_feature)


        if global_cond is not None and lang_global_conditioning_context is not None and traj_global_conditioning_context is not None:
            # pdb.set_trace()
            global_feature = torch.cat([
                global_feature, lang_global_conditioning_context, traj_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None and lang_global_conditioning_context is not None:
            global_feature = torch.cat([
                global_feature, lang_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None and traj_global_conditioning_context is not None:
            global_feature = torch.cat([
                global_feature, traj_global_conditioning_context, global_cond
            ], axis=-1)
        elif global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        

        first_down_module = self.down_modules[0][0]
        # pdb.set_trace()
        first_down_module_cond_encoder = first_down_module.cond_encoder
        x = first_down_module_cond_encoder(global_feature)

        
        return x
