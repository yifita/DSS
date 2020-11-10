from typing import Optional
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from ..utils import get_class_from_string

class Texture(nn.Module):
    def __init__(self, dim=3, c_dim=256,
                 n_layers=4, hidden_size=512, out_dim=3,
                 use_normal=True,
                 use_view_direction=True,
                 pos_encoding=True,
                 num_frequencies=4,
                 **kwargs):
        super().__init__()
        self.pos_encoding = pos_encoding
        self.use_normal = use_normal
        self.dim = dim
        self.c_dim = c_dim
        if use_normal:
            dim += self.dim
        if use_view_direction:
            if pos_encoding:
                self.positional_encoding = PosEncodingNeRF(dim=self.dim,
                                                           num_frequencies=num_frequencies,
                                                           sidelength=kwargs.get(
                                                               'sidelength', None),
                                                           fn_samples=kwargs.get(
                                                               'fn_samples', None),
                                                           use_nyquist=kwargs.get('use_nyquist', True))
                dim = self.positional_encoding.out_dim + dim
            else:
                dim += self.dim

        layers = [nn.Linear(dim + c_dim, hidden_size), nn.ReLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, out_dim))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, xyz, normals=None, view_direction=None, c=None, **kwargs):
        """
        Args:
            x: (N,*,self.dim) point positions
            n: (N,*,self.dim) point normals
            c: (N,*,c_dim) latent code
        """
        assert(xyz.shape[-1] == self.dim)
        inp = xyz
        if self.use_normal:
            normals = torch.nn.functional.normalize(normals, dim=-1)
            inp = torch.cat([normals, inp], dim=-1)
        if view_direction is not None:
            if self.pos_encoding:
                inp = torch.cat(
                    [self.positional_encoding(view_direction), inp], dim=-1)
            else:
                inp = torch.cat([view_direction, inp], dim=-1)

        if c is not None:
            assert(inp.ndim == c.ndim)
            inp = torch.cat([c, inp], dim=-1)

        point_colors = self.layers(inp)
        # shift to [0, 1]
        point_colors = (point_colors + 1) / 2.0
        return point_colors


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, dim, out_dim, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.dim = dim
        self.linear = nn.Linear(dim, out_dim, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.dim,
                                            1 / self.dim)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.dim) / self.omega_0,
                                            np.sqrt(6 / self.dim) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, dim: int, hidden_size: int = 256,
                 n_layers: int = 3, out_dim: int = 1,
                 outermost_linear: bool = True, c_dim: int = 256,
                 first_omega_0: float = 30,
                 hidden_omega_0: float = 30.,
                 activation: Optional[str] = None,
                 **kwargs,
                 ):
        """
        Args:
            dim: first input dimension
            hidden_size: intermediate feature dimension
            n_layers: number of hidden layers (total number of layers = n_layers + 2)
            out_dim: last output dimension
            outermost_linear: use linear layer as the last layer instead of sine layer
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.c_dim = c_dim

        self.net = []
        self.net.append(SineLayer(dim + c_dim, hidden_size,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(n_layers):
            self.net.append(SineLayer(hidden_size, hidden_size,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, out_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / hidden_omega_0,
                                             np.sqrt(6 / hidden_size) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_size, out_dim,
                                      is_first=False, omega_0=hidden_omega_0))

        self.use_activation = activation is not None

        if self.use_activation:
            self.last_activation = get_class_from_string(activation)()
            self.net.append(self.last_activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, c=None, **kwargs):
        """
        Args:
            coords: input coordinates (N, *, dim)
            c (tensor): code (N, 1, c_dim)
        Returns:
            (N, *, out_dim)
        """
        # coords = coords.clone().detach().requires_grad_(
        #     True)  # allows to take derivative w.r.t. input
        if c is not None and c.numel() > 0:
            assert(coords.ndim == c.ndim)
            coords = torch.cat([c, coords], dim=-1)

        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''
        Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!
        '''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, dim, num_frequencies=None, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.dim = dim

        self.num_frequencies = num_frequencies
        if self.num_frequencies is None:
            if self.dim == 3:
                self.num_frequencies = num_frequencies or 10
            elif self.dim == 2:
                self.num_frequencies = num_frequencies
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(
                        min(sidelength[0], sidelength[1]))
            elif self.dim == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(
                        fn_samples)

        self.out_dim = 2 * dim * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        """
        coords: (N, *, D)
        """
        shp = coords.shape

        coords_pos_enc = []
        for i in range(self.num_frequencies):
            for j in range(self.dim):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc.append(torch.cat((sin, cos), dim=-1))

        coords_pos_enc = torch.cat(coords_pos_enc, dim=-1)
        return coords_pos_enc.reshape(shp[:-1] + (self.out_dim,))


class SDF(nn.Module):
    '''
    Based on: https://github.com/facebookresearch/DeepSDF
    and https://github.com/matanatz/SAL/blob/master/code/model/network.py

    If activation is defined , last layer is x = activation(x) + 1.0 * x
    '''

    def __init__(
        self,
        dim: int = 3,
        out_dim: int = 1,
        c_dim: int = 128,
        hidden_size: int = 512,
        n_layers: int = 8,
        dropout=list(),
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all: bool = False,
        activation: Optional[str] = None,
        latent_dropout=False,
        pos_encoding=False,
        num_frequencies=10,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.c_dim = c_dim
        self.out_dim = out_dim
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.positional_encoding = PosEncodingNeRF(dim=dim,
                                                       num_frequencies=num_frequencies,
                                                       sidelength=kwargs.get(
                                                           'sidelength', None),
                                                       fn_samples=kwargs.get(
                                                           'fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True))
            dim = self.positional_encoding.out_dim + self.dim

        dims = [c_dim + dim] + [hidden_size] * n_layers + [out_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            lin = nn.Linear(dims[l], out_dim)

            if (l in dropout):
                p = 1 - dropout_prob
            else:
                p = 1.0

            if l == self.num_layers - 2:
                # last weight layer
                torch.nn.init.normal_(
                    lin.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(p * dims[l]), std=0.000001)
                # c = -r
                torch.nn.init.constant_(lin.bias, -1.0)
            elif l == 0:
                # inputs: [code, positional_code, xyz], set the weights for code and positional
                # code to 0
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(
                    lin.weight, 0.0, np.sqrt(2) / np.sqrt(p * out_dim))
                lin.weight[..., :-self.dim] = 0
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(
                    lin.weight, 0.0, np.sqrt(2) / np.sqrt(p * out_dim))

            if weight_norm and l in self.norm_layers:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.use_activation = activation is not None

        if self.use_activation:
            self.last_activation = get_class_from_string(activation)()
        # self.relu = nn.ReLU()

        self.activation = F.relu

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    def forward(self, input, c=None, only_sdf=True, only_z=False, **kwargs):
        """
        Args:
            input: [N, *, dim]
            c:     [N, 1, c_dim]
        Returns:   [N, *, out_dim]
        """
        if c is not None and c.nelement():
            assert(input.ndim == c.ndim)
            input = torch.cat([c, input], dim=-1)

        xyz = input[..., -self.dim:]
        if self.pos_encoding:
            xyz = torch.cat([self.positional_encoding(xyz), xyz], dim=-1)
            input = torch.cat([input[..., :self.c_dim], xyz], dim=-1)

        if input.shape[-1] > 3 and self.latent_dropout:
            latent_vecs = input[..., :self.c_dim]
            latent_vecs = self.latent_dropout(latent_vecs)
            # latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], -1)
        else:
            x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            # NOTE: Supplement, extend geometric initialization to skip-connections
            if l in self.latent_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob,
                                  training=self.training)

        if self.use_activation:
            x = self.last_activation(x) + 1.0 * x

        if only_sdf:
            x = x[..., :1]
            return x

        if only_z:
            x = x[..., 1:]

        return x


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class Occupancy(nn.Module):
    ''' Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=512, leaky=False, n_blocks=5, out_dim=4, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c=None, only_occupancy=False,
                only_texture=False, **kwargs):
        """
        Args:
            p: (N, *, dim)
            c: (N, 1, c_dim)
        Returns:
            (N, *, out_dim)
        """
        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                assert(p.ndim == c.ndim)
                net_c = self.fc_c[n](c)
                # if p.dim == 3 and net_c.dim == 2:
                #     net_c = net_c.unsqueeze(1)
                # NOTE(yifan) use plus is quite unconventional
                # conditonal batch normalization like spade?
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        if only_occupancy:
            out = out[..., :1]
        elif only_texture:
            out = out[..., 1:4]

        return out


def approximate_gradient(points, network, c=None, h=1e-3, **forward_kwargs):
    ''' Calculates the central difference for points.

    It approximates the derivative at the given points as follows:
        f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

    Args:
        points (tensor): points (N,*,3)
        c (tensor): latent conditioned code c (N, *, C)
        h (float): step size for central difference method
    '''
    n_points, _ = points.shape

    if c is not None:
        c = c.unsqueeze(-2).repeat((1,) * len(c.shape[:-1]) + (6, 1))
        c = c.view(points.shape[0] * 6, -1)

    # calculate steps x + h/2 and x - h/2 for all 3 dimensions
    step = torch.cat([
        torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
        torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
        torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
        torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
        torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
        torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
    ], dim=1).to(points.device) * h / 2
    points_eval = (points.unsqueeze(-2) + step).view(-1, 3)

    # Eval decoder at these points
    f = network.forward(points_eval, c=c, **
                        forward_kwargs)[..., :1].view(n_points, 6)

    # Get approximate derivate as f(x + h/2) - f(x - h/2)
    df_dx = torch.stack([
        (f[:, 0] - f[:, 1]) / h,
        (f[:, 2] - f[:, 3]) / h,
        (f[:, 4] - f[:, 5]) / h,
    ], dim=-1)
    return df_dx


class ResidualSDF(nn.Module):
    def __init__(self, dim: int = 3,
                 out_dim: int = 1,
                 c_dim: int = 128,
                 hidden_size: int = 512,
                 n_layers: int = 8,
                 norm_layers=(),
                 latent_in=(),
                 weight_norm=False,
                 activation: str = None,
                 siren_hidden_size: int = 256,
                 siren_n_layers: int = 3,
                 first_omega_0: float = 30,
                 siren_activation: str= None,
                 hidden_omega_0: float = 30,
                 outermost_linear: bool = True,
                 **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.dim = dim
        self.base = SDF(dim=dim, out_dim=out_dim,
                        c_dim=c_dim, hidden_size=hidden_size,
                        n_layers=n_layers, norm_layers=norm_layers,
                        latent_in=latent_in, weight_norm=weight_norm,
                        xyz_in_all=False, activation=None,
                        latent_dropout=False,
                        pos_encoding=False)
        # self.positional_encoding = PosEncodingNeRF(dim=dim,
        #                                            sidelength=kwargs.get(
        #                                                'sidelength', None),
        #                                            fn_samples=kwargs.get(
        #                                                'fn_samples', None),
        #                                            use_nyquist=kwargs.get('use_nyquist', True))
        # self.residual = SDF(dim=self.positional_encoding.out_dim, out_dim=out_dim,
        #                 c_dim=c_dim, hidden_size=siren_hidden_size,
        #                 n_layers=siren_n_layers, norm_layers=norm_layers,
        #                 latent_in=latent_in, weight_norm=weight_norm,
        #                 xyz_in_all=False, activation=None,
        #                 latent_dropout=False,
        #                 pos_encoding=False)

        self.residual = Siren(dim=dim, hidden_size=siren_hidden_size,
                              n_layers=siren_n_layers, out_dim=1,
                              c_dim=0, outermost_linear=True,
                              first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
                              activation=siren_activation)

        # # change Siren last layer scale
        # last_linear_layer = self.residual.net[-1]
        # assert(isinstance(last_linear_layer, torch.nn.Linear))
        # # original initialization: -np.sqrt(6 / hidden_size) / hidden_omega_0
        # last_linear_layer.weight.data /= 5

    def forward(self, input, c=None, only_base=False, only_residual=False, **kwargs):
        """
        Args:
            input: [N, *, dim]
            c:     [N, 1, c_dim]
            only_base (bool): if true, only return the result from the base net (MLP SDF)
        Returns:   [N, *, out_dim]
        """
        output = self.base(input, c=c, **kwargs)
        if only_base:
            return output
        # input = self.positional_encoding(input)
        # res = self.residual(input, c=c, **kwargs)
        # res = torch.tanh(res) * 0.1
        res = self.residual(input, c=c, **kwargs)
        if only_residual:
            return res
        R = 100
        activation = (1 + R) / (R + torch.exp(output**2 / 0.01))
        output = output + (res * activation.detach())
        return output
