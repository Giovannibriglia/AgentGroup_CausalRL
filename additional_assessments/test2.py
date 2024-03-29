import numpy as np
import global_variables
import os
import json
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

"""
Comparison of Offline and Online Causal Q-Learning in various Test Environments: we conducted experiments 
comparing offline and online causal Q-Learning in identical test settings. For offline causal Q-Learning, we used two 
distinct causal tables:
    1) In the first case, the causal table was derived from an environment where the goal was positioned in the top-right
    corner of the grid. Due to this placement, certain causal relationships, such as "if the goal is on my left,
    then I go left," were impossible to establish. This resulted in an incomplete causal table.
    2) The second causal table was extracted from an environment where the goal was placed in one of the central cells of
    the grid. Here, the 'Goal_Nearby_Agent' feature was observable across all possible values.

Therefore, we have three kinds of algorithms:
    1) Offline causal Q-Learning with incomplete causal knowledge
    2) Offline causal Q-Learning with complete causal knowledge
    3) Online causal Q-Learning

We evaluated the performance differences between these algorithms across three 4x4 grid environments, each containing
two enemies:
    1) The goal was situated in the bottom-left corner in the first environment.
    2) In the second environment, the goal was adjacent to the right edge of the grid but not in any corner.
    3) The third environment featured the goal positioned in one of the central cells.

"""


# TODO: think about how we can pass and in which format the predefined environment

