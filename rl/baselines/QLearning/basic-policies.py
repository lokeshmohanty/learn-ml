import torch
from pathlib import Path
import pandas

from base import interact

def random_policy(env, _obs):
    return env.action_space.sample()

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    filename_rp = 'data/rp.pt'

    if Path(filename_rp).is_file():
        rp = torch.load(filename_rp)
    else:
        rp = interact(random_policy, render_mode="human")
        torch.save(rp, filename_rp)

    # print(f"Random policy: {torch.mean(rp, dtype=torch.float)}")
    for d in rp:
        print(d)
