from plots import plot_av_rew_steps

algorithms = ['QL', 'CQL4', 'DeepQNetwork_Mod', 'CausalDeepQNetwork_Mod']
n_games = 1
n_episodes = 1500
dir = 'Results/Grid/RandEnAct/2En/5x5'

plot_av_rew_steps(dir, algorithms, n_games, n_episodes)