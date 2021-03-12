from typing import Optional, Union
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict, namedtuple
from ..utils import get_class_from_string

"""
SIREN
https://github.com/vsitzmann/siren

DVR

Lipman
"""

""" value / uncertainty, value can further be sdf / latent / rgb """

_fields = ('value', 'sdf', 'uncertainty', 'latent', 'rgb', 'occupancy')
_net_output = namedtuple("Result", _fields,
                         defaults=(None,) * len(_fields))


def _validate_out_dims(out_dims: dict) -> None:
    for k in out_dims.keys():
        if k not in _fields:
            raise ValueError(
                'out_dims contains at least one invalid key {} (valid keys are: {})'.format(k, _fields))
        if k in ('sdf', 'occupancy', 'uncertainty'):
            assert(out_dims[k] == 1)
        elif k == 'rgb':
            assert(out_dims[k] == 3)
        else:
            pass


class BaseModel(nn.Module):
    def __init__(self, out_dims: Union[dict, int]):
        super().__init__()
        _validate_out_dims(out_dims)
        self.out_dim = sum(out_dims.values())
        self._out_dims = tuple(out_dims.values())
        self._out_fields = tuple(out_dims.keys())
        use_uncertainty = ('uncertainty' in out_dims)
        self.use_uncertainty = use_uncertainty

    def _parse_output(self, forward_result: torch.Tensor, scale_rgb=False) -> _net_output:
        out_dict = dict(zip(self._out_fields, torch.split(
            forward_result, self._out_dims, dim=-1)))
        if scale_rgb and 'rgb' in out_dict:
            # shift to [0, 1] or use sigmoid
            out_dict['rgb'] = (out_dict['rgb'] + 1) / 2.0
        return _net_output(**out_dict)


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


class Siren(BaseModel):
    def __init__(self, dim: int, hidden_size: int = 256,
                 n_layers: int = 3, out_dims: dict = OrderedDict(sdf=1),
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
            activation: for the sdf value
        """
        super().__init__(out_dims)
        self.dim = dim
        self.c_dim = c_dim

        self.net = []
        self.net.append(SineLayer(dim + c_dim, hidden_size,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(n_layers):
            self.net.append(SineLayer(hidden_size, hidden_size,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, self.out_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / hidden_omega_0,
                                             np.sqrt(6 / hidden_size) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_size, self.out_dim,
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

        if self.use_activation and isinstance(self.last_activation, torch.nn.Tanh):
            results = self._parse_output(output, scale_rgb=True)
        elif self.use_activation and isinstance(self.last_activation, torch.nn.Sigmoid):
            results = self._parse_output(output, scale_rgb=False)
        else:
            results = self._parse_output(output, scale_rgb=False)
            if results.rgb is not None:
                results = results._replace(rgb=torch.sigmoid(results.rgb))

        return results


""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class SDF(BaseModel):
    '''
    Based on: https://github.com/facebookresearch/DeepSDF
    and https://github.com/matanatz/SAL/blob/master/code/model/network.py
    '''
    def __init__(
        self,
        dim: int = 3,
        out_dims: dict = dict(sdf=1),
        c_dim: int = 0,
        hidden_size: int = 512,
        n_layers: int = 8,
        bias: float = 0.6,
        weight_norm: bool = True,
        skip_in=(4,),
        num_frequencies=6,
        **kwargs
    ):
        super().__init__(out_dims)
        dims = [dim] + [hidden_size] * n_layers + [self.out_dim]

        self.embed_fn = None
        if num_frequencies > 0:
            embed_fn, input_ch = get_embedder(num_frequencies)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.dim = dims[0]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(
                    np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            elif num_frequencies > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(
                    lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif num_frequencies > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(
                    lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(
                    lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, c=None, **kwargs):

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input
        if c is not None and c.numel() > 0:
            assert(x.ndim == c.ndim)
            x = torch.cat([c, x], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        results = self._parse_output(x, scale_rgb=False)
        if results.rgb is not None:
            results = results._replace(rgb=torch.sigmoid(results.rgb))

        return results


class RenderingNetwork(BaseModel):
    def __init__(
        self,
        dim: int = 9,
        out_dims: dict = dict(rgb=3),
        c_dim: int = 256,
        hidden_size: int = 512,
        n_layers: int = 4,
        weight_norm: bool = True,
        num_frequencies=4,
        **kwargs
    ):
        super().__init__(out_dims)
        # self.mode = mode
        self.c_dim = c_dim
        dims = [dim + c_dim] + [hidden_size]*n_layers + [self.out_dim]

        self.embed_fn = None
        if num_frequencies > 0:
            embedview_fn, input_ch = get_embedder(num_frequencies)
            self.embed_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.dim = dims[0]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, c=None, **kwargs):
        if c is not None and c.numel() > 0:
            assert(x.ndim == c.ndim)
            x = torch.cat([c, x], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        results = self._parse_output(x, scale_rgb=True)
        return results

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


class Occupancy(BaseModel):
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
                 hidden_size=512, leaky=False, n_blocks=5,
                 out_dims=OrderedDict(occupancy=1, rgb=3),
                 **kwargs):
        super().__init__(out_dims)
        self.c_dim = c_dim
        self.dim = dim
        self.n_blocks = n_blocks

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, self.out_dim)

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

    def forward(self, p, c=None, only_occupancy=False, only_texture=False, **kwargs):
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

        results = self._parse_output(out, scale_rgb=False)
        if results.rgb is not None:
            results = results._replace(rgb=torch.sigmoid(results.rgb))
        if only_occupancy:
            return results.sdf
        if only_texture:
            return results.rgb
        return results


def approximate_gradient(points, network, c=None, h=1e-3, requires_grad=False, **forward_kwargs):
    ''' Calculates the central difference for points.

    It approximates the derivative at the given points as follows:
        f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

    Args:
        points (tensor): points (N,*,3)
        c (tensor): latent conditioned code c (N, *, C)
        h (float): step size for central difference method
    '''
    n_points, _ = points.shape

    if c is not None and c.nelement() > 0:
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
                        forward_kwargs).sdf.view(n_points, 6)
    if not requires_grad:
        f = f.detach()

    # Get approximate derivate as f(x + h/2) - f(x - h/2)
    df_dx = torch.stack([
        (f[:, 0] - f[:, 1]) / h,
        (f[:, 2] - f[:, 3]) / h,
        (f[:, 4] - f[:, 5]) / h,
    ], dim=-1)

    return df_dx


class ResidualSDF(BaseModel):
    def __init__(self, dim: int = 3,
                 out_dims: dict = OrderedDict(sdf=1),
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
                 siren_activation: str = None,
                 hidden_omega_0: float = 30,
                 outermost_linear: bool = True,
                 **kwargs):
        super().__init__(out_dims)
        self.dim = dim
        self.base = SDF(dim=dim, out_dims=out_dims,
                        c_dim=c_dim, hidden_size=hidden_size,
                        n_layers=n_layers, norm_layers=norm_layers,
                        latent_in=latent_in, weight_norm=weight_norm,
                        xyz_in_all=False, activation=activation,
                        latent_dropout=False,
                        pos_encoding=False)

        self.residual = Siren(dim=dim, hidden_size=siren_hidden_size,
                              n_layers=siren_n_layers, out_dims=out_dims,
                              c_dim=0, outermost_linear=True,
                              first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
                              activation=siren_activation)

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

        # TODO: last layer of base?
        res = self.residual(input, c=c, **kwargs)

        if only_residual:
            return self._parse_output(res)

        R = 100
        activation = (1 + R) / (R + torch.exp(output.sdf**2 / 0.01))
        # activation = (output.abs() < 0.1)
        sdf = output.sdf + activation.detach() * res.sdf
        return output._replace(sdf=sdf)
