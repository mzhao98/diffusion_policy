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
        dcon = 10
        # context_encoder = nn.Sequential(
        #     # SinusoidalPosEmb(dcon),
        #     nn.Linear(dcon, dcon * 4),
        #     nn.Mish(),
        #     nn.Linear(dcon * 4, dsed),
        # )
        # context_encoder = nn.Embedding(2, dsed)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.context_encoder = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True)
        # get dimension of the hidden states
        dbert = 768
        


        # cond_dim = dsed
        cond_dim = dsed + dbert
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

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

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, context_cond=None, **kwargs):
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
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # create empty tensor to store the embeddings
        # global_conditioning_context = torch.zeros((context_cond.shape[0], 768), device=sample.device)
        # pdb.set_trace()
        global_conditioning_context = context_cond
        # convert to float
        global_conditioning_context = global_conditioning_context.float()

        # print("global feature shape: ", global_feature.shape)
        # print("global conditioning context shape: ", global_conditioning_context.shape)
        # pdb.set_trace()
        # for i in range(context_cond.shape[0]):
            
        #     # text_1 = "Replace me by any text you'd like."
        #     if context_cond[i] % 2 == 0:
        #         text_label = 'right'
        #     else:
        #         text_label = 'left'
        #     # pdb.set_trace()
        #     # if text_label == 'right':
        #     #     pdb.set_trace()
        #     # tokenized_text = self.tokenizer.tokenize(text_label)
        #     # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #     # segments_ids = [1] * len(tokenized_text)
        #     tokenized_inputs = self.tokenizer(text_label, padding=True, truncation=True, return_tensors="pt")
        #     # to tensor on device
        #     tokenized_inputs = tokenized_inputs.to(sample.device)

            

        #     with torch.no_grad(): 
        #         outputs = self.context_encoder(**tokenized_inputs)

        #     # Extract the embeddings (using the [CLS] token representation)
        #     embeddings = outputs.last_hidden_state[:, 0, :]

        #     # store the embeddings
        #     global_conditioning_context[i] = embeddings[0]
            # pdb.set_trace()

            # encoded_text = tokenized_inputs.to(sample.device)
            # output = self.context_encoder(encoded_text)

        # 2. conditioning on context
        # UNCOMMENT BELOW FOR EMBEDDDINGS WITHOUT BERT
        # conditioning_context = context_cond
        # # squeeze to 1d
        # if conditioning_context is not None:
        #     conditioning_context = conditioning_context.squeeze(-1)
        #     if not torch.is_tensor(conditioning_context):
        #         # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #         conditioning_context = torch.tensor([conditioning_context], dtype=torch.long, device=sample.device)
        #     elif torch.is_tensor(conditioning_context) and len(conditioning_context.shape) == 0:
        #         conditioning_context = conditioning_context[None].to(sample.device)
        #     # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        #     conditioning_context = conditioning_context.expand(sample.shape[0])
        #     # pdb.set_trace()
        #     # global_conditioning_context = self.context_encoder(conditioning_context.float())
        #     global_conditioning_context = self.context_encoder(conditioning_context.long())

        # else:
        #     global_conditioning_context = torch.zeros_like(global_feature)


        # pdb.set_trace()
        # set conditioning context to 0s if context_cond is even and 1s if it is odd
        # pdb.set_trace()
        # global_cond is [54, 40], global_conditioning_context is [54, 256]
        # if context_cond is even, set global_conditioning_context at dim 1 to all 0
        # if context_cond is odd, set global_conditioning_context at dim 1 to all 1
        # global_conditioning_context = global_conditioning_context * 0
        # for i in range(global_conditioning_context.shape[0]):
        #     global_conditioning_context[i] = global_conditioning_context[i] + context_cond[i] % 2
        
            
        




        if global_cond is not None and global_conditioning_context is not None:
            global_feature = torch.cat([
                global_feature, global_conditioning_context, global_cond
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

