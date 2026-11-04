"""InKAN actor and critic, preserving the original Classic Control architecture."""
from torch import nn
from inkan import KANLayer
from stable_baselines3.common.policies import ActorCriticPolicy


class ClassicExtractor(nn.Module):
    def __init__(self, inputs, widths, grid_size):
        super().__init__()
        dimensions = [inputs] + list(widths)
        def network():
            return nn.Sequential(*[KANLayer(a, b, grid_size=grid_size, compile_basis=False)
                                   for a, b in zip(dimensions[:-1], dimensions[1:])])
        self.policy_net, self.value_net = network(), network()
        self.latent_dim_pi = self.latent_dim_vf = widths[-1]

    def forward_actor(self, x):
        return self.policy_net(x)

    def forward_critic(self, x):
        return self.value_net(x)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class ClassicKANPolicy(ActorCriticPolicy):
    def __init__(self, *args, widths=(2, 2), grid_size=5, **kwargs):
        self.widths, self.grid_size = list(widths), grid_size
        kwargs['ortho_init'] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ClassicExtractor(self.features_dim, self.widths, self.grid_size)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'widths': self.widths, 'grid_size': self.grid_size}
