import itertools
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd # type: ignore
from typing import List, Tuple, Callable, cast

class PositionalEncoding(torch.nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()
        self.freqs: torch.Tensor
        self.register_buffer("freqs", 2**torch.arange(0, n_freqs) * torch.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[...,None] * self.freqs
        x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return x.flatten(-2)

# From https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
class TruncatedExponential(Function):  # pylint: disable=abstract-method
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))

truncated_exp: Callable = TruncatedExponential.apply

"""Vanilla NeRF"""

class VanillaFeatureMLP(torch.nn.Module):
    def __init__(self, n_freqs: int, hidden_features: List[int]):
        super().__init__()
        in_features = n_freqs * 2 * 3
        self.net = torch.nn.Sequential(
            PositionalEncoding(n_freqs),
            torch.nn.Linear(in_features, hidden_features[0]),
            *[torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_features[k], hidden_features[k+1]),
            ) for k in range(len(hidden_features)-1)],
        )
        self.feature_dim = hidden_features[-1]

    def forward(self, x):
        return self.net(x)

class VanillaOpacityDecoder(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1),
            torch.nn.Softplus()
        )

    # TODO: activation function for density?
    def forward(self, x):
        return self.net(x)

class VanillaColorDecoder(torch.nn.Module):
    def __init__(self, n_freqs: int, in_features: int, hidden_features: List[int]):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs)
        self.total_features = in_features + n_freqs * 2 * 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.total_features, hidden_features[0]),
            *[torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_features[k], hidden_features[k+1]),
            ) for k in range(len(hidden_features)-1)],
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features[-1], 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.pe(rays_d), features], -1)
        return self.net(x)

"""K-Planes https://arxiv.org/abs/2301.10241"""

class KPlanesFeaturePlane(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = 8,
        resolution: Tuple[int, int] = (128, 128),
        init: Callable = torch.nn.init.uniform_
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.plane = torch.nn.Parameter(torch.empty(1, feature_dim, *resolution))
        init(self.plane)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 2)"""
        new_shape = [*x.size()[:-1],self.feature_dim]
        output = torch.nn.functional.grid_sample(
            self.plane,
            x.view(1,-1,1,2),
            align_corners=True 
        ).squeeze().transpose(0,-1).contiguous()
        return output.view(new_shape)

    def loss_tv(self) -> torch.Tensor:
        tv_x = torch.nn.functional.mse_loss(self.plane[:, :, 1:, :], self.plane[:, :, :-1, :])
        tv_y = torch.nn.functional.mse_loss(self.plane[:, :, :, 1:], self.plane[:, :, :, :-1])
        return tv_x + tv_y

    def loss_l1(self) -> torch.Tensor:
        return torch.mean(torch.abs(self.plane))

class KPlanesFeatureField(torch.nn.Module):
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.planes = torch.nn.ModuleList([
            torch.nn.ModuleList([
                KPlanesFeaturePlane(feature_dim, resolution=(128,128)),
                KPlanesFeaturePlane(feature_dim, resolution=(128,128)),
                KPlanesFeaturePlane(feature_dim, resolution=(128,128)),
            ]),
            torch.nn.ModuleList([
                KPlanesFeaturePlane(feature_dim, resolution=(256,256)),
                KPlanesFeaturePlane(feature_dim, resolution=(256,256)),
                KPlanesFeaturePlane(feature_dim, resolution=(256,256)),
            ]),
            torch.nn.ModuleList([
                KPlanesFeaturePlane(feature_dim, resolution=(512,512)),
                KPlanesFeaturePlane(feature_dim, resolution=(512,512)),
                KPlanesFeaturePlane(feature_dim, resolution=(512,512)),
            ])
        ])
        # pairs of coordinates that will be used to compute plane feature *in that order*
        # check the order if you want to have specific resolution for a given dimension (e.g. t in the paper)
        self.dimension_pairs = list(itertools.combinations(range(3), 2))
        self.feature_dim = 32 * len(self.planes)

        for plane_scale in self.planes:
            assert isinstance(plane_scale, torch.nn.ModuleList)
            assert len(plane_scale) == len(self.dimension_pairs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 3)
        returns (..., self.feature_dim)"""
        features = []
        for plane_scale in self.planes:
            current_scale_features = 1.
            for (i, j), plane in zip(self.dimension_pairs, plane_scale): # type: ignore
                current_scale_features *= plane(x[..., (i,j)])
            features.append(current_scale_features)
        return torch.cat(features, -1) # type: ignore

    def loss_tv(self) -> torch.Tensor:
        loss = 0.
        count = 0
        for plane_scale in self.planes:
            for plane in plane_scale: # type: ignore
                loss += plane.loss_tv()
                count += 1
        return cast(torch.Tensor, loss) / count

    def loss_l1(self) -> torch.Tensor:
        loss = 0.
        count = 0
        for plane_scale in self.planes:
            for plane in plane_scale: # type: ignore
                loss += plane.loss_l1()
                count += 1
        return cast(torch.Tensor, loss) / count

class KPlanesExplicitOpacityDecoder(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
        )
        self.activation = lambda x: truncated_exp(x - 1.)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.sum(features * self.net(features), -1, keepdim=True)
        return self.activation(x)

class KPlanesExplicitColorDecoder(torch.nn.Module):
    def __init__(self, feature_dim, n_freqs = 8, hidden_dim = 128):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs)
        self.feature_dim = feature_dim
        in_dim = feature_dim + n_freqs * 2 * 3 + 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3 * feature_dim),
        )

    def forward(self, features: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.pe(rays_d), rays_d, features], -1)
        x = self.net(x).view(-1, 3, self.feature_dim)
        output = torch.sum(features.unsqueeze(-2) * x, -1)
        return torch.sigmoid(output)

class KPlanesHybridOpacityDecoder(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1)
        )
        self.activation = lambda x: truncated_exp(x - 1.)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(features))


class KPlanesHybridColorDecoder(torch.nn.Module):
    def __init__(self, feature_dim, n_freqs = 8, hidden_dim = 64):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs)
        self.feature_dim = feature_dim
        in_dim = feature_dim + n_freqs * 2 * 3 + 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.pe(rays_d), rays_d, features], -1)
        return self.net(x)