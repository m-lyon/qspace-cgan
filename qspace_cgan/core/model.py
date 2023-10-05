'''Generator and Discriminator models for Q-Space CGAN'''
from typing import List
import torch
from torch.nn import functional as F

import pytorch_lightning as pl


class AdaptiveInstanceNorm2d(torch.nn.Module):
    '''Adaptive Instance Norm 2D'''

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        '''Runs forward pass'''
        assert (
            self.weight is not None and self.bias is not None
        ), 'Assign weight and bias before calling AdaIN!'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)  # type: ignore
        running_var = self.running_var.repeat(b)  # type: ignore

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])


class LayerNorm(torch.nn.Module):
    '''Layer Normalization'''

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = torch.nn.parameter.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = torch.nn.parameter.Parameter(torch.zeros(num_features))

    def forward(self, x):
        '''Runs forward pass'''
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Conv2dBlock(torch.nn.Module):
    '''2D Convolutional Block'''

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm='none',
        activation='relu',
        pad_type='zero',
    ):
        super().__init__()
        self.use_bias = True

        # Initialize padding
        if pad_type == 'zero':
            self.pad = torch.nn.ZeroPad2d(padding)
        if pad_type == 'reflect':
            self.pad = torch.nn.ReflectionPad2d(padding)
        else:
            raise AttributeError(f'Unsupported padding type: {pad_type}')

        # Initialize normalization
        if norm == 'none':
            self.norm = None
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(output_dim)
        else:
            raise AttributeError(f'Unsupported normalization: {norm}')

        # initialize activation
        if activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise AttributeError(f'Unsupported activation: {activation}')

        # initialize convolution
        if norm == 'sn':
            pass
            # self.conv = SpectralNorm(
            #     torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
            # )
        else:
            self.conv = torch.nn.Conv2d(
                input_dim, output_dim, kernel_size, stride, bias=self.use_bias
            )

    def forward(self, x):
        '''Forward pass'''
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(torch.nn.Module):
    '''Residual Block'''

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type),
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type),
        )

    def forward(self, x):
        '''Forward pass'''
        residual = x
        out = self.layers(x)
        out = out + residual
        return out


class ResBlocks(torch.nn.Module):
    '''Collection of Residual Blocks'''

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


class LinearBlock(torch.nn.Module):
    '''Linear Block'''

    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super().__init__()
        use_bias = True
        # initialize fully connected layer
        # if norm == 'sn':
        #     self.layer = SpectralNorm(torch.nn.Linear(input_dim, output_dim, bias=use_bias))
        # else:
        self.layer = torch.nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        if norm in ('none', 'sn'):
            self.norm = None
        else:
            raise AttributeError(f'Unsupported normalization: {norm}')

        # initialize activation
        if activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise AttributeError(f'Unsupported activation: {activation}')

    def forward(self, x):
        '''Forward pass'''
        out = self.layer(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(torch.nn.Module):
    '''Multi-Layer Perceptron'''

    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super().__init__()
        layers: List[torch.nn.Module] = [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for _ in range(n_blk - 2):
            layers.append(LinearBlock(dim, dim, norm=norm, activation=activ))
        self.layers = torch.nn.Sequential(*layers)
        self.final = LinearBlock(dim, output_dim, norm='none', activation='none')

    def forward(self, x):
        '''Forward pass'''
        feature = self.layers(x.view(x.size(0), -1))
        return self.final(feature)


class Encoder(torch.nn.Module):
    '''Q-Space Generator Encoder'''

    def __init__(self, n_downsample, n_res, input_dim, dim):
        super().__init__()
        layers = []
        layers.append(
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm='adain', activation='relu', pad_type='reflect'
            )
        )
        # Downsampling blocks
        for _ in range(n_downsample):
            layers.append(
                Conv2dBlock(
                    dim, 2 * dim, 4, 2, 1, norm='adain', activation='relu', pad_type='reflect'
                )
            )
            dim *= 2
        # rRsidual blocks
        layers.append(ResBlocks(n_res, dim, norm='adain', activation='relu', pad_type='reflect'))
        self.layers = torch.nn.Sequential(*layers)
        self.output_dim = dim

    def forward(self, x):
        '''Runs forward pass'''
        return self.layers(x)


class Decoder(torch.nn.Module):
    '''Q-Space Generator Decoder'''

    def __init__(self, n_upsample, n_res, dim, output_dim):
        super().__init__()

        # AdaIN residual blocks
        layers: List[torch.nn.Module] = [ResBlocks(n_res, dim, 'adain', 'none', pad_type='reflect')]
        # upsampling blocks
        for _ in range(n_upsample):
            layers.extend(
                [
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    Conv2dBlock(
                        dim, dim // 2, 5, 1, 2, norm='ln', activation='relu', pad_type='reflect'
                    ),
                ]
            )
            dim //= 2
        # use reflection padding in the last conv layer
        layers.append(
            Conv2dBlock(
                dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type='reflect'
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''Runs forward pass'''
        return self.layers(x)


class Generator(torch.nn.Module):
    '''Q-Space CGAN Generator'''

    def __init__(self):
        super().__init__()

        self.encoder = Encoder(2, 4, 3, 64)
        self.decoder = Decoder(2, 4, self.encoder.output_dim, 1)

        # MLP to generate AdaIN parameters
        self.mlp_encoder = MLP(4, self.get_num_adain_params(self.encoder), 256, 3)
        self.mlp_decoder = MLP(4, self.get_num_adain_params(self.decoder), 256, 3)

    def assign_adain_params(self, adain_params, model):
        '''Assigns AdaIN parameters to the AdaIN layers in the model'''
        # assign the adain_params to the AdaIN layers in model
        for module in model.modules():
            if isinstance(module, AdaptiveInstanceNorm2d):
                mean = adain_params[:, : module.num_features]
                std = adain_params[:, module.num_features : 2 * module.num_features]
                module.bias = mean.contiguous().view(-1)
                module.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * module.num_features:
                    adain_params = adain_params[:, 2 * module.num_features :]

    def get_num_adain_params(self, model):
        '''Returns the number of AdaIN parameters needed by the model'''
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for module in model.modules():
            if isinstance(module, AdaptiveInstanceNorm2d):
                num_adain_params += 2 * module.num_features
        return num_adain_params

    def forward(self, b0_vol, bvec):
        '''Runs forward pass'''
        adain_params_enc = self.mlp_encoder(bvec)
        self.assign_adain_params(adain_params_enc, self.encoder)
        content = self.encoder(b0_vol)
        adain_params_dec = self.mlp_decoder(bvec)
        self.assign_adain_params(adain_params_dec, self.decoder)
        dwi = self.decoder(content)
        return dwi
