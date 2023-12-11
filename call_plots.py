import plots

algorithms = ['QL_EpsGreedy', 'QL_BoltzmannMachine', 'QL_ThompsonSampling', 'QL_SoftAnn']
n_games = 5
n_episodes = 1000
dir = 'Comparison2_QLearning_DifferentPolicy/Grid/RandEnAct/1Enem/5x5'

# plots.plot_av_rew_steps(dir, algorithms, n_games, n_episodes)
plots.plot_av_computation_time(dir, algorithms, n_games, n_episodes, 5, 5, 1)