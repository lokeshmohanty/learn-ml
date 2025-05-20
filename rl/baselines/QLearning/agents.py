import torch as t
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from clearml import Task

from base import interact

task = Task.init(
    project_name="PhD Thesis/General Algorithms",
    task_name="QLearning-SARSA",
    tags=["env:frozenlake-v1", "rl"],
    auto_connect_frameworks={"tensorboard": True, "matplotlib": True, "pytorch": True},
)


class QLearning:
    def __init__(self, alpha=0.8, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma
        self.q = t.zeros(16, 4, dtype=t.float)

    def train(self, eps=0.1, **kwargs):
        self.eps = eps
        self.train_data = interact(self.policy, learner=self.update, **kwargs)

    def update(self, env, s, a, r, s1):
        td = r + self.gamma * self.q[s1].max() - self.q[s, a]
        self.q[s, a] = self.q[s, a] + self.alpha * td

    def policy(self, env, s):
        if t.rand(1)[0] < self.eps or t.all(self.q[s] == self.q[s][0]):
            return env.action_space.sample()
        return self.q[s].argmax().item()

    def evaluate(self, **kwargs):
        self.eps = 0
        return interact(self.policy, **kwargs)


class Sarsa(QLearning):
    def update(self, env, s, a, r, s1):
        a1 = self.policy(env, s1)
        td = r + self.gamma * self.q[s1, a1] - self.q[s, a]
        self.q[s, a] = self.q[s, a] + self.alpha * td


def plot(q, env):
    map_size = int(sqrt(q.shape[0]))
    q_val_max = q.max(1).values.reshape(map_size, map_size)
    q_best_action = q.argmax(1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    q_directions = [["" for _ in range(map_size)] for _ in range(map_size)]
    eps = t.finfo(t.float).eps  # Minimum float number on the machine
    for s, v in enumerate(q):
        s1, s2 = s // map_size, s % map_size
        a = q_best_action[s1, s2]
        if v[a] > eps:
            q_directions[s1][s2] = directions[a.item()]

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    env.reset()
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        q_val_max,
        annot=q_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"q_{map_size}x{map_size}.png"
    fig.savefig(Path("../results") / img_title, bbox_inches="tight")
    plt.show()


def plot_rewards(rewards_df, steps_df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")

    fig.tight_layout()
    img_title = "rewards.png"
    fig.savefig(Path("../results") / img_title, bbox_inches="tight")
    plt.show()


# plot_steps_and_rewards(res_all, st_all)

## Main
if __name__ == "__main__":
    Path("../results").mkdir(exist_ok=True)
    filename_ql = "../results/ql.pt"
    filename_sarsa = "../results/sarsa.pt"

    useCache = False
    if "useCache" in sys.argv[1:]:
        useCache = True
    if "sarsa" in sys.argv[1:]:
        if useCache and Path(filename_sarsa).is_file():
            model = t.load(filename_sarsa)
        else:
            model = Sarsa()
            qs = []
            for i in range(20):
                model.q = t.zeros(16, 4, dtype=t.float)
                model.train(episodes=2000, run=i + 1, max_runs=20)
                qs.append(model.q)
            model.q = t.stack(qs).mean(0)
            t.save(model, filename_sarsa)
    else:
        if useCache and Path(filename_ql).is_file():
            model = t.load(filename_ql)
        else:
            model = QLearning()
            qs = []
            for i in range(20):
                model.q = t.zeros(16, 4, dtype=t.float)
                model.train(episodes=2000, run=i + 1, max_runs=20)
                qs.append(model.q)
            model.q = t.stack(qs).mean(0)
            t.save(model, filename_ql)

    _, env = model.evaluate(episodes=0, return_env=True)
    plot(model.q, env)
