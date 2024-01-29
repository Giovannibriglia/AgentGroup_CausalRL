import numpy as np
import os

current_directory = os.getcwd()
print("Current directory:", current_directory)

try:
   print(np.load('home/gbriglia/CausalRL/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
except:
    try:
        print(np.load('gbriglia/CausalRL/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
    except:
        try:
            print(np.load('CausalRL/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
        except:
            try:
                print(np.load('Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
            except:
                print('nothing')
