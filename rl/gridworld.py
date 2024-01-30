# Example 3.5: Gridworld

import jax.numpy as jnp
import numpy as np
from jax import random, jit
from typing import TypeAlias
import math
from rich.console import Console
import numpy as np

# north -> up, south -> down, east -> right, west -> left
env = { "actions" : ["north", "south", "east", "west"]
      , "states" : [(i, j) for i in range(5) for j in range(5)]
      }

Reward: TypeAlias = float
State: TypeAlias = tuple[int, int]
def act(env, state, action) -> tuple[Reward, State]:
    if state == (0, 1):
        return 10, (4, 1)
    if state == (0, 3):
        return 5, (2, 3)
    if action == env["actions"][0] and state[0] != 0:
        return 0, (state[0] - 1, state[1])
    if action == env["actions"][1] and state[0] != 4:
        return 0, (state[0] + 1, state[1])
    if action == env["actions"][2] and state[1] != 4:
        return 0, (state[0], state[1] + 1)
    if action == env["actions"][3] and state[1] != 0:
        return 0, (state[0], state[1] - 1)
    return -1, state


Action: TypeAlias = str
Key: TypeAlias = list[int]
def policy(env, state, key) -> tuple[Action, Key]:
    old, new = random.split(key)
    index = min(len(env["actions"]) - 1,
                math.floor(random.uniform(old) * len(env["actions"])))
    return env["actions"][index], new
    
def compute_value(env, gamma=0.9) -> list[list[int]]:    
    m = len(env["states"])
    coeff = np.zeros((m, m))
    b = np.zeros((m, 1))
    for i, j in env["states"]:
        coeff[i*5 + j][i*5 + j] = -1
        for a in env["actions"]:
            print(act(env, (i,j), a), a)
            r, s = act(env, (i, j), a)
            coeff[i*5 + j][s[0]*5 + s[1]] += 0.25 * gamma
            b[i*5 + j] += 0.25 * r
    val = np.linalg.pinv(coeff).dot(b)
    return [[val[5 * i + j][0] for j in range(5)] for i in range(5)]

console = Console()
console.print(np.asarray(compute_value(env)))
