# AI4REALNET D1.3 RL deployment plug-in

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

The plugin loads aircraft, sector and restricted area data (either historical or synthetic), and sends observations to the previously trained AI models from the BlueSky-Gym environments, which in turn control the speed and heading of the aircraft. This set-up allows to test RL models on real world scenarios, enabling the measurement of KPIs.

### References
Groot, D. Janthony and Leto, Giulia and Vlaskin, Aleksandr and Moec, Adam and Ellerbroek, Joost. "BlueSky-Gym: Reinforcement Learning Environment for Air Traffic Applications", 14th SESAR Innovation Days. https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_021%20final.pdf (2024).

## Features
The following environments are supported:
- StaticObstacleEnv-v0
- StaticObstacleCREnv-v0
- StaticObstacleSectorEnv-v0
- StaticObstacleSectorCREnv-v0
- CentralisedStaticObstacleEnv-v0
- CentralisedStaticObstacleCREnv-v0
- CentralisedStaticObstacleSectorCREnv-v0

## Installation
The following dependencies are required

```bash
pip install .
pip install '.[full]'
pip install stable-baselines3 
```
## Usage
Run BlueSky using `python BlueSky.py`

Choose which environment you would like to test. 
Load one of the scenarios files called 'ai4realnet_deploy_RL.scn' contained in scenario/ai4realnet_deploy_RL/{env-name}/

Testing scenario with real world data are available on AI4REALNET GitLab.

## Project Structure

Each supported BlueSky-Gym environment has its own dedicated plug-in. The main files of the plug-ins are:

bluesky/
└── plugins/
    ├── ai4realnet_deploy_RL_centralised_static_obstacle.py
    ├── ai4realnet_deploy_RL_centralised_static_obstacle_cr.py
    ├── ai4realnet_deploy_RL_centralised_static_obstacle_sector_cr.py
    ├── ai4realnet_deploy_RL_static_obstacle_cr.py
    ├── ai4realnet_deploy_RL_static_obstacle_sector_cr.py
    ├── ai4realnet_deploy_RL_static_obstacle_sector.py
    └── ai4realnet_deploy_RL_static_obstacle.py


Helper functions and models can be found in the following tree (both environment specific and common):
bluesky/
└── plugins/
    └── ai4realnet_deploy_RL_tools/
        ├── CentralisedStaticObstacleCREnv
        ├── CentralisedStaticObstacleEnv
        ├── CentralisedStaticObstacleSectorCREnv
        ├── models/
        ├── StaticObstacleCREnv
        ├── StaticObstacleEnv
        ├── StaticObstacleSectorCREnv
        ├── StaticObstacleSectorEnv
        ├── functions.py
        └── READ_ME.md

The scenario files containing the data on which the models are tested can be found in the following:
scenario/
└── ai4realnet_deploy_RL/
    ├── CentralisedStaticObstacleCREnv-v0/
    ├── CentralisedStaticObstacleEnv-v0/
    ├── CentralisedStaticObstacleSectorCREnv-v0/
    ├── StaticObstacleCREnv-v0/
    ├── StaticObstacleEnv-v0/
    ├── StaticObstacleSectorCREnv-v0/
    ├── StaticObstacleSectorEnv-v0/
    └── sector.scn

Each of the folders above contains:
└── {env-name}/
   ├── flights-sample.scn
   ├── restricted_areas-sample.scn
   └── ai4realnet_deploy_RL.scn

To run real world scenario, substitute the files 'flights-sample.scn' and 'restricted_areas-sample.scn' with files containing such data.

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