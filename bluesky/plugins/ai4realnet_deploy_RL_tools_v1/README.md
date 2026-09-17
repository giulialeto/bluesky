# AI4REALNET RL deployment plug-in with randomly generated scenarios and perturbations v1

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
- StaticObstacleSectorCREnv-v0

## Installation
The following dependencies are required

```bash
pip install .
pip install '.[full]'
pip install stable-baselines3 
```
## Usage
Run BlueSky using `python BlueSky.py`

Load the scenarios called `ai4realnet_deploy_RL_randomised_scenario_loading_v1.scn` contained in `scenario/ai4realnet_deploy_RL_randomised_scen_loading_v1-2/StaticObstacleSectorCREnv-v0/`

## Project Structure

The main file of the plug-in is:
```
bluesky/
└── plugins/
    └── ai4realnet_deploy_RL_static_obstacle_sector_cr_randomised_scenario_v1.py
```

Helper functions and models can be found in the following tree:
```
bluesky/
└── plugins/
    └── ai4realnet_deploy_RL_tools_v1/
    │   ├── StaticObstacleSectorCREnv
    │   ├── functions.py
    │   └── READ_ME.md
    └── ai4realnet_deploy_RL_models/
```

The scenario files containing the data on which the models are tested can be found in the following:
```
scenario/
└── ai4realnet_deploy_RL_randomised_scen_loading_v1-2/
    ├── StaticObstacleSectorCREnv-v0/ai4realnet_deploy_RL_randomised_scenario_loading_v1.scn
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
