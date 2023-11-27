import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from scipy.ndimage import gaussian_filter1d

dir_results = 'Results2\Grid\RandEnAct\\1En\\5x5'
algorithms = ['CQL4', 'DeepQNetwork', 'CausalDeepQNetwork', 'DeepQNetwork_Mod']


targetPattern = fr"{dir_results}\*.npy"
directories = glob.glob(targetPattern)
print(directories)
data_names = [os.path.basename(directories[s]) for s in range(len(directories))]


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
fig.suptitle(f'Performance comparison - Grid 5x5 - 1 enemy', fontsize=15)

for alg in algorithms:
    filename_rewards = [s for s in directories if f'{alg}_rewards' in s]
    filename_steps = [s for s in directories if f'{alg}_steps' in s]

    av_rewards = np.array([0*len(filename_rewards[0])])
    av_steps = np.array([0*len(filename_rewards[0])])
    for n_game in range(len(filename_rewards)):
        av_rewards += np.load(filename_rewards[n_game])
        av_steps += np.load(filename_steps[n_game])

    av_rewards = av_rewards / len(filename_rewards)
    av_steps = av_steps / len(filename_rewards)

    x = np.arange(0, len(av_rewards), 1)
    ax1.plot(x, gaussian_filter1d(av_rewards, 1), label=f'{alg} = {round(np.mean(av_rewards), 3)}')
    confidence_interval_rew = np.std(av_rewards)
    ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.1)
    ax1.set_title('Average reward on episode steps')
    ax1.legend(fontsize='x-small')

    ax2.plot(x, gaussian_filter1d(av_steps, 1))
    ax2.set_yscale('log')
    ax2.set_title('Steps needed to complete the episode')
    ax2.set_xlabel('Episode', fontsize=12)

plt.show()