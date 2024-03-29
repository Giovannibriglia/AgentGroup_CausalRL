# Uguali al test 1 ma con ambiente toroidale, prendi gli stessi ambienti del test 1
import numpy as np
import global_variables
import os
import json
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

"""
We develop this assessment to validate our hypothesis concerning the extraction of causal relationships. Our 
hypothesis posits that there arises an issue when an agent, operating in both offline and online contexts, 
selects an action that leads it to collide with a wall or boundary of the environment. While this aspect does not 
directly impact reinforcement learning (RL) algorithms, it significantly affects the causal discovery process. Here, 
the action value itself is not the stopping action, but rather, the variables DeltaX and DeltaY both have a 
value of zero. Consequently, this introduces erroneous information into both the causal graph and the causal table.

We believe this problem is environment-dependent. In wider environments, this issue exists but holds lesser 
significance, whereas in smaller environments, it becomes considerably more relevant.

As a result of our simulations, we present final causal graphs and causal tables across various environmental 
configurations; all simulations are conducted with 1 agent, 1 enemy and 1 goal in a grid-like world environments of 
different sizes. A high number of training episodes has been selected for this assessment to ensure thorough 
exploration of the entire environment by the agent and to guarantee the comprehensive development of the causal 
table, capturing all possible dependencies within the specified episodes.

Additionally, if we consider a square grid with dimensions rows * cols, the number of 'central' cells is (rows-2)^2. 
For clarity, consider the following example:
    - grid 3x3 --> 1 central cell, 8 boundary cells 
    - grid 4x4 --> 4 central cells, 12 boundary cells 
    - grid 8x8 --> 36 central cells, 28 boundary cells
"""