import jax
import jax.numpy as jnp
import equinox as eqx

class DQN(eqx.Module):

    def __init__(self, n_observations, n_actions):
        self.layer1 = eqx.Linear(n_observations, 128)
        self.layer2 = eqx.Linear(128, 128)
        self.layer3 = eqx.Linear(128, n_actions)
