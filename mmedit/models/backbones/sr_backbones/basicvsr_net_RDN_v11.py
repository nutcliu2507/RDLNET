# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from easydict import EasyDict


@BACKBONES.register_module()
class BasicVSRNet_RDN_v11(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, n_RDB=5, n_RDB_c=3, with_tsa=False, spynet_pretrained=None, SMSR_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.n_RDB = n_RDB
        self.n_RDB_c = n_RDB_c
        self.with_tsa = with_tsa

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.training = True

        # propagation branches

        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        # self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels, mid_channels, 8)
        # self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels, mid_channels, 8)


        smsr_arg = EasyDict({
            'n_feats': self.mid_channels,
            'scale':[4],
            'rgb_range':0,
            'n_colors':3
        })
        self.SMM_tau_step = -5.98e-6
        # self.backward_SMSR_SMM = SMSR_SMM(smsr_arg, num_smm=5, training=self.training, load_path=SMSR_pretrained)
        # self.forward_SMSR_SMM = SMSR_SMM(smsr_arg, num_smm=5, training=self.training, load_path=SMSR_pretrained)
        # self.SMSR_SMM = SMSR_SMM(smsr_arg, num_smm=5, training=self.training, load_path=SMSR_pretrained)

        kernel_size = 3
        smsr_arg = EasyDict({
            'n_feats': self.mid_channels,
            'scale':[4],
            'rgb_range':0,
            'n_colors':3
        })
        growRate0 = mid_channels
        growRate = mid_channels
        nConvLayers = self.n_RDB_c
        # self.cnn_64 = nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True)
        # self.first_RDN = RDN(1, growRate0, growRate, nConvLayers)
        self.backward_RDN = RDN(3, growRate0, growRate, nConvLayers)
        self.forward_RDN = RDN(3, growRate0, growRate, nConvLayers)
        # self.reduction = nn.Conv2d(self.mid_channels*3, self.mid_channels, 3, 1, 1, bias=True)
        # self.last_RDN = RDN(5, growRate0, growRate, nConvLayers)
        

        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=3,
                center_frame_idx=1)
        else:
            self.fusion = nn.Conv2d(self.mid_channels*3, self.mid_channels, 3, 1, 1, bias=True)


        

        # upsample
        # self.reconstruction = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, 5)
        self.upconv1 = nn.Conv2d(self.mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation function
        self.mish = nn.Mish(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False


    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lqs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        logger = get_root_logger()
        n, t, c, h, w = lqs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')


        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lqs)
        if flows_forward is None:
            flows_forward = flows_backward.flip(1)

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        backward_frame_idx = frame_idx[::-1]
        backward_flow_idx = backward_frame_idx

        feats = {}
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        # feats_ = self.lrelu(self.cnn_64(lqs.view(-1, c, h, w)))
        # feats_ = self.first_RDN(feats_)
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
        # mapping_idx >> 為了鏡像資料
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        feats['backward_prop'] = []
        feats['frontward_prop'] = []


        # backward-time propgation
        out_l = []
        feat_prop = flows_backward.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            feat_current = feats['spatial'][i]
            if i < t-1:
                # backward_flow_idx_i = backward_flow_idx[i]
                # logger.info(f'backward_flow_idx_i:{backward_flow_idx_i}')
                flow_n1 = flows_backward[:, i, :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i < t-2:
                    feat_n2 = feats['backward_prop'][-2]
                    flow_n2 = flows_backward[:, i + 1, :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            else:
                cond_n1 = feat_prop
                cond_n2 = torch.zeros_like(cond_n1)
            # logger.info(f'feat_current_size:{feat_current.size()}')
            # logger.info(f'cond_n1_size:{cond_n1.size()}')
            # logger.info(f'cond_n2_size:{cond_n2.size()}')
            '''
            feat_current_size:torch.Size([2, 64, 64, 64])
            cond_n1_size:torch.Size([2, 64, 64, 64])
            cond_n2_size:torch.Size([2, 64, 64, 64])
            feat_prop_size > 2, 192, 64, 64
            '''
            
            
            # logger.info(f'feat_prop_size:{feat_prop.size()}')
            # feat_prop_size:torch.Size([2, 192, 64, 64])
            if self.with_tsa:
                cond_n1 = cond_n1.unsqueeze(1)
                feat_current_2 = feat_current.unsqueeze(1)
                cond_n2 = cond_n2.unsqueeze(1)
                feat_prop = torch.cat([cond_n1, feat_current_2, cond_n2], dim=1)
                feat_prop = self.fusion(feat_prop).squeeze(1)
            else:
                feat_prop = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = self.fusion(feat_prop)
            # feat_prop = self.lrelu(feat_prop)
            feat_prop = self.mish(feat_prop)
            # feat_prop = self.backward_resblocks(feat_prop)
            feat_prop = self.backward_RDN(feat_prop)
            feats['backward_prop'].append(feat_prop)
        feats['backward_prop'] = feats['backward_prop'][::-1]

        # forward-time propagation and upsampling
        feat_prop = lqs.new_zeros(n, self.mid_channels, h, w)
        for i in range(0, t):
            lqs_i = lqs[:, i, :, :, :]
            feat_current = feats['backward_prop'][i]
            if i > 0:
                flow_n1 = flows_forward[:, i-1 , :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats['frontward_prop'][-2]
                    flow_n2 = flows_forward[:, i-2, :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            else:
                cond_n1 = feat_prop
                cond_n2 = torch.zeros_like(cond_n1)
            if self.with_tsa:
                cond_n1 = cond_n1.unsqueeze(1)
                feat_current_2 = feat_current.unsqueeze(1)
                cond_n2 = cond_n2.unsqueeze(1)
                feat_prop = torch.cat([cond_n1, feat_current_2, cond_n2], dim=1)
                feat_prop = self.fusion(feat_prop).squeeze(1)
            else:
                feat_prop = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = self.fusion(feat_prop)
            # feat_prop = self.lrelu(feat_prop)
            feat_prop = self.mish(feat_prop)
            # feat_prop = self.forward_resblocks(feat_prop)
            feat_prop = self.forward_RDN(feat_prop)
            feats['frontward_prop'].append(feat_prop)

            # upsample
            out = feat_prop
            # logger.info(f'feat_current_size:{feat_current.size()}')
            # logger.info(f'feat_prop:{feat_prop.size()}')
            # feats_spatial_i_size =feats['spatial'][i].size()
            # logger.info(f'feats_spatial_i:{feats_spatial_i_size}')
        
            # out = torch.cat([feat_prop, feat_current, feats['spatial'][i]], dim=1)
            # out = self.lrelu(self.reduction(out))
            # out = self.last_RDN(self.mish(self.reduction(out)))
            # out = self.reconstruction(out)
            
            '''
            if self.training:
                out, sparsity= self.SMSR_SMM(out)
            else:
                out = self.SMSR_SMM(out)
            '''

            out = self.mish(self.pixel_shuffle(self.upconv1(out)))
            out = self.mish(self.pixel_shuffle(self.upconv2(out)))
            out = self.mish(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(lqs_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)
            # out_sparsity[i] = all_sparsity

        return torch.stack(out_l, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


    def _set_tau(self, current_iter):
        # self.backward_SMSR_SMM._set_tau(current_iter, self.SMM_tau_step)
        # self.forward_SMSR_SMM._set_tau(current_iter, self.SMM_tau_step)
        self.SMSR_SMM._set_tau(current_iter, self.tau_step)

    def _set_test(self):
        self.training = False

    def _set_train(self):
        self.training = True

    def _prepare(self):
        # self.backward_SMSR_SMM._prepare()
        # self.forward_SMSR_SMM._prepare()
        self.SMSR_SMM._prepare()

    def change_value(self, x, out_min, out_max, in_min, in_max):
        step = (out_max-out_min)/(in_max-in_min)
        y = out_min + x*step
        return y



class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            # nn.Mish(inplace=True)
            # nn.Mish()
            nn.ReLU()
            # nn.LeakyReLU(negative_slope=0.1)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, n_RDB, growRate0, growRate, nConvLayers):
        super(RDN, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        kSize = 3
        self.D = n_RDB
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):
        fea = x
        RDBs_out = []
        for i in range(self.D):
            fea = self.RDBs[i](fea)
            RDBs_out.append(fea)

        fea = self.GFF(torch.cat(RDBs_out,1))
        x = x + fea
        return x


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob

        # fusion
        feat = self.feat_fusion(aligned_feat)

        # spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat



def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, training, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4):
        super(SMB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training = training
        self.n_layers = n_layers

        self.tau = 1
        self.relu = nn.ReLU(True)

        # channels mask
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))

        # body
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers-1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)

        # collect
        self.collect = nn.Conv2d(out_channels*self.n_layers, out_channels, 1, 1, 0)

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self):
        # channel mask
        ch_mask = self.ch_mask.softmax(3).round()
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers):
            if i == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))
            else:
                self.d_in_num.append(int(ch_mask[0, :, i-1, 0].sum(0)))
                self.s_in_num.append(int(ch_mask[0, :, i-1, 1].sum(0)))
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        for i in range(self.n_layers):
            if i == 0:
                kernel_s.append([])
                if self.d_out_num[i] > 0:
                    kernel_d2d.append(self.body[i].weight[ch_mask[0, :, i, 0]==1, ...].view(self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.s_out_num[i] > 0:
                    kernel_d2s.append(self.body[i].weight[ch_mask[0, :, i, 1]==1, ...].view(self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
            else:
                if self.d_in_num[i] > 0 and self.d_out_num[i] > 0:
                    kernel_d2d.append(
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i-1, 0] == 1, ...].view(self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[i] > 0 and self.s_out_num[i] > 0:
                    kernel_d2s.append(
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i-1, 0] == 1, ...].view(self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
                if self.s_in_num[i] > 0:
                    kernel_s.append(torch.cat((
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...],
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...]),
                        0).view(self.d_out_num[i]+self.s_out_num[i], -1))
                else:
                    kernel_s.append([])

        # the last 1x1 conv
        ch_mask = ch_mask[0, ...].transpose(1, 0).contiguous().view(-1, 2)
        self.d_in_num.append(int(ch_mask[:, 0].sum(0)))
        self.s_in_num.append(int(ch_mask[:, 1].sum(0)))
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(self.collect.weight[:, ch_mask[..., 0] == 1, ...].squeeze())
        kernel_d2s.append([])
        kernel_s.append(self.collect.weight[:, ch_mask[..., 1] == 1, ...].squeeze())

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias

    def _generate_indices(self):
        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze())

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)

    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))
                else:
                    fea_col = fea_dense.view(self.d_in_num[index], -1)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel_d2s[index], self._mask_select(fea_dense, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = torch.mm(self.kernel_s[index], F.pad(fea_sparse, [1,0,0,0])[:, self.idx_s2s].view(self.s_in_num[index] * k * k, -1))

        # fusion
        if self.d_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_d2d[0, :, self.h_idx_1x1, self.w_idx_1x1] += fea_s2ds[:self.d_out_num[index], :]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
                fea_d[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds[:self.d_out_num[index], :]
        else:
            fea_d = None

        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_s = fea_d2s + fea_s2ds[ -self.s_out_num[index]:, :]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[-self.s_out_num[index]:, :]
        else:
            fea_s = None

        # add bias (bias is only used in the last 1x1 conv in our SMB for simplicity)
        if index == 4:
            fea_d += self.bias.view(1, -1, 1, 1)

        return fea_d, fea_s


    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature (B, C ,H, W) ;
        x[1]: spatial mask (B, 1, H, W)
        '''

        if self.training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

            out = []
            fea = x[0]
            for i in range(self.n_layers):
                if i == 0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea * ch_mask[:, :, i:i + 1, :1]
                else:
                    fea_d = self.body[i](fea * ch_mask[:, :, i - 1:i, :1])
                    fea_s = self.body[i](fea * ch_mask[:, :, i - 1:i, 1:])
                    fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + \
                          fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
                fea = self.relu(fea)
                out.append(fea)

            out = self.collect(torch.cat(out, 1))

            return out, ch_mask

        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            for i in range(self.n_layers):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k=3, index=i)
                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))

            # 1x1 conv
            fea_dense = torch.cat(fea_dense, 1)
            fea_sparse = torch.cat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index=self.n_layers)

            return out


    def forward_orig(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature (B, C ,H, W) ;
        x[1]: spatial mask (B, 1, H, W)
        '''
        spa_mask = x[1]
        ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

        out = []
        fea = x[0]
        for i in range(self.n_layers):
            if i == 0:
                fea = self.body[i](fea)
                fea = fea * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea * ch_mask[:, :, i:i + 1, :1]
            else:
                fea_d = self.body[i](fea * ch_mask[:, :, i:i + 1, :1])
                fea_s = self.body[i](fea * ch_mask[:, :, i:i + 1, 1:])
                fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + \
                        fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
            fea = self.relu(fea)
            out.append(fea)

        out = self.collect(torch.cat(out, 1))

        return out, ch_mask


class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, training, kernel_size=3, stride=1, padding=1, bias=False):
        super(SMM, self).__init__()
        self.training = training

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // 4, 2, 3, 2, 1, output_padding=1),
        )

        # body
        self.body = SMB(in_channels, out_channels, training, kernel_size, stride, padding, bias, n_layers=4)

        # CA layer
        self.ca = CALayer(out_channels)

        self.tau = 1

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        if self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = gumbel_softmax(spa_mask, 1, self.tau)

            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x

            return out, spa_mask[:, 1:, ...], ch_mask


        if not self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = (spa_mask[:, 1:, ...] > spa_mask[:, :1, ...]).float()

            out = self.body([x, spa_mask])
            out = self.ca(out) + x
            return out

    def forward_test(self, x):
        spa_mask = self.spa_mask(x)
        spa_mask = gumbel_softmax(spa_mask, 1, self.tau)

        out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
        out = self.ca(out) + x

        return out, spa_mask[:, 1:, ...], ch_mask


class SMSR_SMM(nn.Module):
    def __init__(self, args, num_smm=5, training=True, load_path=None):
        super(SMSR_SMM, self).__init__()
        n_feats = args.n_feats
        self.num_smm = num_smm
        kernel_size = 3
        self.scale = int(args.scale[0])
        self.training = training

        # define body module
        self.modules_body = [SMM(n_feats, n_feats, kernel_size) \
                        for _ in range(self.num_smm)]

        # define collect module
        self.collect = nn.Sequential(
            nn.Conv2d(n_feats*self.num_smm, n_feats, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        self.body = nn.Sequential(*self.modules_body)
        if load_path:
            # self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])
            kwargs = {'map_location': lambda storage, loc: storage}
            self.load_state_dict(torch.load(load_path, **kwargs), strict=False)
            
            for param in self.parameters():
                param.requires_grad=False
            

    def forward(self, x):
        if self.training:
            sparsity = []
            out_fea = []
            fea = x
            for i in range(self.num_smm):
                fea, _spa_mask, _ch_mask = self.body[i](fea)
                out_fea.append(fea)
                sparsity.append(_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1))
            out_fea = self.collect(torch.cat(out_fea, 1)) + x
            sparsity = torch.cat(sparsity, 0)
            return [out_fea, sparsity]

        if not self.training:
            out_fea = []
            fea = x
            for i in range(self.num_smm):
                fea = self.body[i](fea)
                out_fea.append(fea)
            out_fea = self.collect(torch.cat(out_fea, 1)) + x
            return out_fea

    def _set_tau(self, current_iter, tau_step):
        current_iter_value = current_iter*tau_step + 0.998
        tau = max(current_iter_value, 0.4)
        for SMM in self.modules_body:
            SMM._update_tau(tau)

    def _prepare(self):
        for SMM in self.modules_body:
            SMM.body._prepare()

    def change_value(self, x, out_min, out_max, in_min, in_max):
        step = (out_max-out_min)/(in_max-in_min)
        y = out_min + x*step
        return y



