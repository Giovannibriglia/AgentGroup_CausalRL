import numpy as np
import os

try:
    print(np.load('/home/gbriglia/AgentGroup_CausalRL/Results_Comparison123/Grid/RandEnAct/2Enem/5x5/env_game1.npy'))
except:
    pass

current_directory = os.getcwd()
print("Current directory:", current_directory)

try:
   print(np.load(f'{current_directory}/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
except:
    try:
        print(np.load('/gbriglia/AgentGroup_CausalRL/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
    except:
        try:
            print(np.load('/AgentGroup_CausalRL/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
        except:
            try:
                print(np.load('/Results_Comparison123/Grid/RandEnAct/5Enem/5x5/env_game5.npy'))
            except:
                print('nothing')

folder_path = os.getenv('FOLDER_PATH')
print(folder_path)
folder_path = os.path.join(os.path.dirname(__file__), 'folder_name')
print(folder_path)