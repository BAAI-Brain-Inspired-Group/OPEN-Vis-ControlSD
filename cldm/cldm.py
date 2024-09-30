import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import numpy as np
import random
from einops import rearrange, repeat,reduce
from torchvision import transforms

# sys.path.append('/home/chenzhiqiang/code/hypercolumn')
from hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class HyperColumnLGN(nn.Module):
    def __init__(self,key=0,hypercond=[0],size=512,restore_ckpt = './hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
        super().__init__()
        ckpt = torch.load(restore_ckpt)
        hc = Column_trans_rot_lgn(ckpt['arg'])
        hc.load_state_dict(ckpt['state_dict'], strict=False)
        self.lgn_ende = hc.lgn_ende[0].eval()
        self.lgn_ende.train = disabled_train
        for param in self.lgn_ende.parameters():
            param.requires_grad = False

        self.resize = transforms.Resize(size)
        if size is 128:
            self.pad = nn.ConstantPad2d((1,1,1,1),0.)
        else:
            self.pad = nn.Identity()
        

        # self.groups = [[2,3],[0,1,4,8,9,15],[5,6,7,10,12],[11,13,14]]
        self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14],[5],[12],[2],[3]]
        # self.groups = [[2],[3],[5],[12]]
            
        # self.p = [1.,0.5,0.25,0.125]
        self.p = [0. for i in range(len(self.groups))]
        self.p[0] = 1.

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None

    # def forward(self,x):
    #     s = x.size()
    #     r = torch.zeros(1,16,1,1).to(x.device)

    #     if self.cond is None:
    #         c = [i for i in range(len(self.groups))]
    #         random.shuffle(c)
    #         # print(self.groups[c[0]])
    #         pa = random.random()
    #         for i in range(4):
    #             if pa < self.p[i]:
    #                 for j in self.groups[c[i]]:
    #                     r[:,j,:,:] = 1.
    #     else:
    #         for i in self.cond:
    #             for j in self.groups[i]:
    #                 r[:,j,:,:] = 1

    #     r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
    #     return self.lgn_ende(self.norm(x))*r
    
    def forward(self,x):
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,16,1,1).to(x.device)

        if self.cond is None:
            c = [i for i in range(len(self.groups))]
            random.shuffle(c)
            # print(self.groups[c[0]])
            pa = random.random()
            for i in range(len(self.groups)):
                if pa < self.p[i]:
                    for j in self.groups[c[i]]:
                        r[:,j,:,:] = 1.
        else:
            for i in self.cond:
                for j in self.groups[i]:
                    r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        out = self.lgn_ende(self.norm(x))*r
        # print('out_conv:',out.size())
        out = self.pad(self.lgn_ende.deconv(out))
        # out = self.lgn_ende.deconv(self.lgn_ende(self.norm(x))*r)
        # print('out:',out.size())
        return out
    
    def deconv(self,x):
        x = self.resize(x)
        s = x.size()
        r = torch.zeros(1,16,1,1).to(x.device)
        if self.cond is not None:
            for i in self.cond:
                for j in self.groups[i]:
                    r[:,j,:,:] = 1

        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        conv = self.lgn_ende(self.norm(x))*r
        deconv = self.lgn_ende.deconv(conv)
        deconv = (deconv - reduce(deconv,'b c h w -> b c () ()', 'min'))/(reduce(deconv,'b c h w -> b c () ()', 'max')-reduce(deconv,'b c h w -> b c () ()', 'min'))*2.-1.

        return deconv
    
class Canny(nn.Module):
    def __init__(self,para=None):
        super().__init__()
    def forward(self,x):
        return x
    def deconv(self,x):
        return x


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            hyperconfig=None,
            stride=[2,2,2],
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.hypercolumn = instantiate_from_config(hyperconfig)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # self.input_hint_block = TimestepEmbedSequential(
        #     conv_nd(dims, hint_channels, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 32, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 96, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )

        # self.input_hint_block_new = TimestepEmbedSequential(
        #     HyperColumnLGN(),
        #     # nn.SiLU(),
        #     conv_nd(dims, 64, 64, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 64, 128, 3, padding=1,stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 128, 128, 3, padding=1,stride=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 128, 256, 3, padding=1,stride=1),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )

        self.input_hint_block_new = TimestepEmbedSequential(
            # HyperColumnLGN(hypercond=hypercond),
            # Canny(),
            self.hypercolumn,
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=stride[0]),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=stride[1]),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=stride[2]),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # print('time_embed:',self.time_embed)

        # print('ControlNet foward emb:',emb.size(),'hint:',hint.size(),'context:',context.size())

        guided_hint = self.input_hint_block_new(hint, emb, context)
        # print(self.input_hint_block_new)
        # print('ControlNet foward emb:',emb.size(),'hint:',hint.size(),'context:',context.size(),'guided_hint:',guided_hint.size())

        outs = []

        h = x.type(self.dtype)
        # print('h:',h.size())
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                # print('h:',h.size())
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,control_step=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.] * 13
        self.control_step = control_step
        # self.control_scales = [1.5] * 4 + [1.]*3 + [0.75]*3 + [0.5]*3
        # self.control_scales = [4.,4.,4.,4.,2.,2.,2.,1.,1.,1.,0.5,0.5,0.5]
        if isinstance(self.control_model.hypercolumn,HyperColumnLGN):
            self.hypercond = control_stage_config['params']['hyperconfig']['params']['hypercond']
        else:
            self.hypercond = [0]
        self.select()
    
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    
    # def validation_step(self, batch, batch_idx):
    #     return super().training_step(batch, batch_idx)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        # print('c:',c)
        
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # print('t:',t)

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            if t[0]<self.control_step:
                control = [c * 0. for c, scale in zip(control, self.control_scales)]
            else:
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.1, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale >= -1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            for slct in self.slct:
                self.control_model.input_hint_block_new[0].cond = slct['cond']
                suffix = slct['suffix']
                print(suffix)
                samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                deconv = self.control_model.input_hint_block_new[0].deconv(c_cat)
                log[f"{suffix}_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
                log[f"{suffix}_deconv_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = deconv
                # self.control_model.input_hint_block_new[0].cond = None

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    @torch.no_grad()
    def select(self):
        # self.slct = [{'cond':[0],'suffix':'0'},{'cond':[1],'suffix':'1'},{'cond':[2],'suffix':'2'},{'cond':[3],'suffix':'3'},{'cond':[0,1],'suffix':'01'},
        #              {'cond':[0,2],'suffix':'02'},{'cond':[0,3],'suffix':'03'},{'cond':[1,2],'suffix':'12'},{'cond':[1,3],'suffix':'13'},{'cond':[2,3],'suffix':'23'},
        #              {'cond':[0,1,2],'suffix':'012'},{'cond':[0,1,3],'suffix':'013'},{'cond':[1,2,3],'suffix':'123'},
        #              {'cond':[0,1,2,3],'suffix':'0123'}]
        self.slct = [{'cond':[i],'suffix':f'{i}'} for i in self.hypercond]
