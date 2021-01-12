import numpy as np

traj_len = 10
xs = np.linspace(1, 10, 10) + 3
ys = np.hstack((np.ones(5), np.linspace(1, 5, 5) * 2)) + 3


exp_trajs = []

for e in range(10):
    mu, sigma = 0, 0.1  # mean and standard deviation
    noise = np.random.normal(mu, sigma, 10)
    xs += noise
    noise = np.random.normal(mu, sigma, 10)
    ys += noise

    ax = xs[1:] - xs[:-1]
    ay = ys[1:] - ys[:-1]

    traj = []
    for i in range(len(ax)):
        action = [ax[i], ay[i]]
        obs = [xs[i], ys[i]]

        trans = {
            "observation": obs,
            "action": action
        }

        traj.append(trans)

    exp_trajs.append(traj)

print(exp_trajs)

import pickle

with open('demo/Lines.pkl', 'wb') as f:
    pickle.dump(exp_trajs, f)
