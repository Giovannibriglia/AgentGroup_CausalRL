# Improving Exploration in RL with Causal Models of Core Environment Dynamics
**Paper**: [Improving Exploration in RL with Causal Models of Core Environment Dynamics]( https://www.ecai2024.eu/calls/main-track)

**Abstract**: @gio copia-incolla l'abstract

**Maintainer**: [Giovanni Briglia](https://github.com/Giovannibriglia)  
**Affiliation**: [Distributed and Pervasive Intelligence Group](https://dipi-unimore.netlify.app/) at [University of Modena and Reggio Emilia](https://www.unimore.it/)  
**Contact**: [stefano.mariani@unimore.it](mailto:stefano.mariani@unimore.it) and [giovanni.briglia@unimore.it](mailto:giovanni.briglia@unimore.it)

![flowchart_resume](flowchart_resume.png)

## Project Structure

```
additional_assessments
   |__init__.py
   |__1_evaluation_for_test1_and_test2.py
   |__1_test1_and_test2
   |__2_evaluation_sensitive_analysis_batch_episodes.py
   |__2_launcher_sensitive_analysis_batch_episodes_for_online_cd.py
   |__3_evaluation_test3.py
   |__3_test3
   |__4.1_launcher_offline_CD_analysis.py
   |__4.2_launcher_offline_CD_analysis_multienv.py
   |__init__.py
Results
   |__Comparison123
   |__Comparison4
   |__OfflineCD_MultiEnv
   |__Sensitive_Analysis_Batch_Episodes
   |__Test1
   |__Test2
   |__Test3
scripts
   |__algorithms
      |__init__py
      |__causal_discovery.py
      |__dqn_agent.py
      |__q_learning_agent.py
      |__random_agent.py
   |__launchers
      |__init__.py
      |__launcher_comparison4_paper.py
      |__launcher_comparison123_paper.py
   |__utils
      |__images_for_render
         |__bowser.png
         |__goal.png
         |__supermario.png
         |__wall.png
      |__init.py
      |__batch_episodes_for_online_cd_values.pkl
      |__dqn_class_and_memory.py
      |__environment.py
      |__exploration_strategies.py
      |__ground_truth_causal_graph.json
      |__ground_truth_causal_graph.png
      |__ground_truth_causal_table.pkl
      |__merge_causal_graphs.py
      |__others.py
      |__seed_values.npy
      |__test_causal_table
      |__train_models.py
   __init__.py
   example.py
   plot_and_tables.py
Videos
   |__Comparison123
   |__Comparison4
__init__.py
experiments_parameters.png
flowchart_resume.png
global_variables.py
LICENSE
README
requirements.txt
setup.py
values_reasons_parameters.png
```

## Installation
1. Create a new python virtual environment with 'python 3.10'
2. Install 'requirements'
   ```
   pip install -r requirements.txt
   ```
3. Install setup
   ```
   python setup.py install
   ```
4. Run test example
   ```
   python3.10 -m scripts/example.py
   ```
## How to Reproduce Paper Results
   For comparison: Vanilla vs Causal Offline vs Causal Online in Grid-like Environments:
   ```
   python setup.py install
   ```
   ```
   python3.10 -m scripts/launchers/launcher_comparison123_paper.py
   ```
   For comparison: With and Without Transfer Learning in Maze-like Environments:
   ```
   python setup.py install
   ```
   ```
   python3.10 -m scripts/launchers/launcher_comparison4_paper.py
   ```

## Parameters
![experiments_parameters](experiments_parameters.png)
<p align="center">
  <img width="460" height="300" src=values_reasons_parameters.png>
</p>

## Develop your Own Extension
Your extension can take various paths:
1) One direction involves modifying the causal discovery algorithms.
2) Another direction entails adding new kinds of agent (currently Q-Learning and DQN have been developed). It's crucial to maintain consistency with the training class by implementing the "__update_Q_or_memory__", "__update_exp_fact__", "__select_action__" and "__return_q_table__" functions. Additionally, in the "global_variables.py" script, you need to include your custom label.
3) The third direction involves testing new environments.

## Citation  
```
@inproceedings{}
```

