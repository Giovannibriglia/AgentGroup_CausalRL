import plots

algorithms = ['QL_EpsilonGreedy',  'QL_SoftmaxAnnealing', 'QL_BoltzmannMachine', 'QL_ThompsonSampling',
              'QL_EpsilonGreedy_Causal', 'QL_SoftmaxAnnealing_Causal',
              'QL_BoltzmannMachine_Causal', 'QL_ThompsonSampling_Causal']
n_games = 3
n_episodes = 100
dir = 'Comparison2_QLearning_DifferentPolicy/Grid/SameEnAct/1Enem/5x5'

plots.plot_av_rew_steps(dir, algorithms, n_games, n_episodes, 5, 5, 1)
plots.plot_av_computation_time(dir, algorithms, n_games, 5, 5, 1)