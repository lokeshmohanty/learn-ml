# %%
## Importing the libraries
import random
import gymnasium as gym
from collections import namedtuple, deque

import torch
from torch import nn, optim
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Training with Experience Replay Memory
# contains transitions: state, action, next state and reward

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# %%
# DQN Algorithm
# Q : State x Action -> reward
# policy(state) = argmax_a Q(state, action)
# update rule -> TD error / MC error
# TD error : Q(s, a) - (r + gamma * max_a Q(s', a))
# Huber loss : (0.5 error^2) if (error <= 1) else (|error| - 0.5)
# Update : Q(s, a) <- Q(s, a) - learning_rate * (error)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

# %%
# Config and Environment
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 5e-3
LR = 1e-4

EPS_START = 0.9
EPS_END = 0.05 
EPS_DECAY = 1000

env = gym.make("CartPole-v1")
state, info = env.reset()
n_actions = env.action_space.n
n_observations = len(state)

# %%
# Network
import math
import gymnasium as gym

policy = DQN(n_observations, n_actions).to(device)
target = DQN(n_observations, n_actions).to(device)
target.load_state_dict(policy.state_dict())

optimizer = optim.AdamW(policy.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                  device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = policy(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # next_state_values[non_final_mask] = policy(non_final_next_states).max(1).values
        next_state_values[non_final_mask] = target(non_final_next_states).max(1).values

    expected_q_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()


steps_done = 0
def epsilon_greedy(env, policy, state):
    global steps_done
    steps_done += 1
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    if torch.rand(1)[0] <= eps:
        # return torch.tensor([env.action_space.sample()])
        return torch.tensor([[env.action_space.sample()]],
                            device=device,
                            dtype=torch.long)
    with torch.no_grad():
        return torch.tensor([[policy(state).argmax()]])
        # return policy(state).max(1).indices.view(1, 1)

# %%
# Post Processing

import matplotlib.pyplot as plt
# from IPython import display

def plot_durations(durations, show_result=False):
    plt.figure(1)
    durations = torch.tensor(durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations.numpy())

    if durations.shape[0] >= 100:
        means = durations.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)            # pause to let plots be updated
    # if show_result:
    #     display.display(plt.gcf())
    #     display.clear_output(wait=True)
    # else:
    #     display.display(plt.gcf())

# %%
# training Loop
from itertools import count

n_episodes = 1000                # 600
durations = []

for i in range(n_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = epsilon_greedy(env, policy, state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation,
                                      dtype=torch.float32,
                                      device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model(memory)

        target_state_dict = target.state_dict()
        policy_state_dict = policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * TAU + (1 - TAU) * target_state_dict[key]
            target.load_state_dict(target_state_dict)

        if done:
            durations.append(t+1)
            # print(durations)
            plot_durations(durations)
            break

print('Complete')
print('durations')
plot_durations(durations, show_result=True)
plt.ioff()
plt.savefig("durations1.png")
plt.show()

