import torch.nn as nn

from .resnet import ResnetBlock
from .cbam import CBAMBlock


class RauneNet(nn.Module):
    """Residual and Attention-driven underwater enhancement Network.
    """
    def __init__(self, input_nc, output_nc, n_blocks, n_down, ngf=64,
                 padding_type='reflect', use_dropout=False, use_att_down=True, use_att_up=False,
                 norm_layer=nn.InstanceNorm2d):
        """Initializes the RAUNE-Net.

        Args:
            input_nc: Number of channels of input images.
            output_nc: Number of chnnels of output images.
            n_blocks: Number of residual blocks.
            n_down: Number of down-sampling blocks.
            ngf: Number of kernels of Conv2d layer in `WRPM`.
            padding_type: Type of padding layer in Residual Block.
            use_dropout: Whether to use dropout.
            use_att_down: Whether to use attention block in down-sampling.
            use_att_up: Whether to use attention block in up-sampling.
            norm_layer: Type of Normalization layer.
        """
        assert (n_blocks >= 0 and n_down >= 0)
        super().__init__()
        use_bias = False if norm_layer else True

        model = []

        # Wide-range Perception Module (WRPM)
        model.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)
        ))

        # Attention Down-sampling Module (ADM)
        for i in range(n_down):
            mult = 2 ** i
            model.append(self._down(ngf*mult, ngf*mult*2, norm_layer=norm_layer, use_att=use_att_down, use_dropout=use_dropout))
        
        # High-level Features Residual Learning Module (HFRLM)
        mult = 2 ** n_down
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias))
        
        # Up-sampling Module (UM)
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model.append(self._up(ngf * mult, int(ngf * mult / 2), use_att=use_att_up, use_dropout=use_dropout))

        # Feature Map Smoothing Module (FMSM) and Tanh Activation Layer
        model.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ))
        
        self.model = nn.Sequential(*model)

    def _down(self, in_channels, out_channels, norm_layer=None, use_att=True, use_dropout=False, dropout_rate=0.5):
        """Attention Down-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            norm_layer: Type of Normalization layer.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        use_bias = False if norm_layer else True
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=use_bias)]
        if norm_layer:
            layers.append(norm_layer(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def _up(self, in_channels, out_channels, use_att=False, use_dropout=False, dropout_rate=0.5):
        """Up-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward function.

        Args:
            input: Input images. Type of `torch.Tensor`.
        """
        return self.model(input)