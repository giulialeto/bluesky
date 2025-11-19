# AI4REALNET D1.4 RL deployment plug-in with randomly generated scenarios and perturbations

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [License](#license)

## Overview
This plug-in complements the release of environments in BlueSky-Gym, a Gymnasium style library for standardized Single-Agent Reinforcement Learning research in Air Traffic Management. BlueSky-Gym provides a set of simplified scenarios, procedurally generated, which are used to train models with reinforcement learning agents (custom or from standard libraries). 

The plugins loads sector data and generates synthetic aircraft and restricted areas. The RL model deployment sends observations to the previously trained AI models from the BlueSky-Gym environments, which in turn control the speed and heading of the aircraft. This set-up allows to test RL models on real world scenarios, enabling the measurement of KPIs.

### References
Groot, D. Janthony and Leto, Giulia and Vlaskin, Aleksandr and Moec, Adam and Ellerbroek, Joost. "BlueSky-Gym: Reinforcement Learning Environment for Air Traffic Applications", 14th SESAR Innovation Days. https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_021%20final.pdf (2024).

## Features
The following environments are supported:
- StaticObstacleEnv-v0
- StaticObstacleCREnv-v0
- StaticObstacleSectorEnv-v0
- StaticObstacleSectorCREnv-v0

Models supported are:
- SAC
- DDPG
- TD3
- PPO

## Installation
The following dependencies are required

```bash
pip install .
pip install '.[full]'
pip install stable-baselines3 
```
## Usage
Run BlueSky using `python BlueSky.py`

Create a scenario file in this template.
```
00:00:00.00>PLUGIN scenario_generator
00:00:00.00>PLUGIN disturbance_generator
# initialize_scenario <N_AC>, <N_OBSTACLES>
00:00:00.00> initialize_scenario 10, 5
00:00:00.00> perturbation weather on
00:00:00.00> perturbation volcanic on
# deploy_RL <ENVIRONMENT>, <ALGORITHM>
00:00:00.00> deploy_RL StaticObstacleEnv-v0 SAC
00:00:00.00>OP
00:00:00.00>DTMULT 5000
00:00:00.00>repeats 10
```
Choose and modify the parameters for this batch of testing: <N_AC>, <N_OBSTACLES>, <ENVIRONMENT>, <ALGORITHM>, <N_SCN>

Start the plugin with 
`plugin deployRL_batch`

run the batch with 
`detached_batch scenario_file_name`


## Project Structure

The main files of the plug-ins are:
```
bluesky/
└── plugins/
    ├── ai4realnet_deploy_RL_batch.py
    ├── ai4realnet_perturbations.py
    └── ai4realnet_random_scenario_generator.py
```

Helper functions and models can be found in the following tree (both environment specific and common):
```
bluesky/
└── plugins/
    ├── ai4realnet_deploy_RL_tools_batch/
    │   ├── __init__.py
    │   ├── constants.py
    │   ├── functions.py
    │   └── READ_ME.md
    └── ai4realnet_deploy_RL_models/
```
The scenario files containing the data on which the models are tested can be found in the following:
```
scenario/
└── ai4realnet_deploy_RL_batch/
    ├── ai4realnet_deploy_RL_batch.scn or your own customised scenario
    ├── config_screen.scn
    ├── sector.scn
    └── generated_scenarios/
```

The above folder `generated_scenarios` contains the scenarios generated while running the batch RL deployment, saved through saveic.
A log for the scenarios can be found in output.

## Authors
- [Giulia Leto](https://github.com/giulialeto)

### Contributors

This project exists thanks to all the people who contribute to [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and to [BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym)

<a href="https://github.com/TUDelft-CNS-ATM/bluesky/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TUDelft-CNS-ATM/bluesky" />
</a>

<a href="https://github.com/TUDelft-CNS-ATM/bluesky-gym/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TUDelft-CNS-ATM/bluesky-gym" />
</a>

## License
MIT License

Copyright (c) 2025 TU Delft
