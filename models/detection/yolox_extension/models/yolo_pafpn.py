"""
Original Yolox PAFPN code with slight modifications
"""
from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...yolox.models.network_blocks import BaseConv, CSPLayer, DWConv
from data.utils.types import BackboneFeatures


class YOLOPAFPN(nn.Module):
    """
    Removed the direct dependency on the backbone.
    """

    def __init__(
            self,
            depth: float = 1.0,
            in_stages: Tuple[int, ...] = (2, 3, 4),
            in_channels: Tuple[int, ...] = (256, 512, 1024),
            depthwise: bool = False,
            act: str = "silu",
            compile_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        assert len(in_stages) == len(in_channels)
        assert len(in_channels) == 3, 'Current implementation only for 3 feature maps'
        self.in_features = in_stages
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')

        ##################################

        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest-exact')
        self.lateral_conv0 = BaseConv(
            in_channels[2], in_channels[1], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            2 * in_channels[1],
            in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            in_channels[1], in_channels[0], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            2 * in_channels[0],
            in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            in_channels[0], in_channels[0], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            2 * in_channels[0],
            in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            in_channels[1], in_channels[1], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            2 * in_channels[1],
            in_channels[2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')
        ##################################

    def forward(self, input: BackboneFeatures):
        """
        Args:
            inputs: Feature maps from backbone

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        features = [input[f] for f in self.in_features]
        x2, x1, x0 = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = th.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = th.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = th.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = th.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
