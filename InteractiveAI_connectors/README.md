# AI4REALNET D1.4 InteractiveAI integration with ATM.UC2

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [License](#license)

## Overview
The release contains a connector between BlueSky and InteractiveAI, developed to enable the interaction between Flow Management Positions (FMPs) and RL agents, in the context of ATM.UC2 (path planning in the presence of restricted areas).

BlueSky is the engine of the simulations. It is used to simulate scenarios, synthetic or historical, in which the aircraft are controlled by an RL agent via BlueSky's plugins (ai4realnet_deploy_RL_batch.py, see Deliverable 1.3). InteractiveAI is the frontend interface with which the FMPs interact. The state of the simulation is pushed as context to InteractiveAI, while discrete events (such as the occurrence of losses of separation or the incursions in restricted areas) trigger notifications that are displayed to the user, with various degrees of urgency. 

The connector supports any RL agents trained with BlueSky-Gym's StaticObstacleCREnv-v1 environment, including the Multi-objective RL agent trained with AI4REALNET's MORL-DOL. The connector also supports the perturbation agent (i.e., the occurrence of weather and volcanic cells).

### References
InteractiveAI: https://github.com/ainetus/InteractiveAI > **TODO:** Add reference here.

MORL: **TODO:** Add reference here.

BlueSky: Hoekstra, Jacco M. and Ellerbroek, Joost."BlueSky ATC Simulator Project: an Open Data and Open Source Approach", 7th International Conference on Research in Air Transportation. https://www.researchgate.net/publication/304490055_BlueSky_ATC_Simulator_Project_an_Open_Data_and_Open_Source_Approach (2016)

BlueSky-Gym: Groot, D. Janthony and Leto, Giulia and Vlaskin, Aleksandr and Moec, Adam and Ellerbroek, Joost. "BlueSky-Gym: Reinforcement Learning Environment for Air Traffic Applications", 14th SESAR Innovation Days. https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_021%20final.pdf (2024).

## Features
The following environments are supported:
- StaticObstacleEnv-v0
- StaticObstacleCREnv-v1

<!-- TODO: - StaticObstacleSectorEnv-v0 -->

<!-- TODO: - StaticObstacleSectorCREnv-v1 -->

Models supported are:
- SAC
- DDPG
- TD3
- PPO

Information notifications in InteractiveAI are triggered:
- when an aircraft enters/leaves the sector
- when a weather/volcanic cell is cleared


Medium priority Alerts in InteractiveAI are triggered:
- when an weather/volcanic cell is detected
<!-- TODO: - when a conflict is detected within 5 minutes? -->


High priority Alerts in InteractiveAI are triggered:
- when a loss of separation occurs
- when an aircraft intrudes a restricted area, a weather cell or a volcanic ash cell.

## Installation and Usage

1. Run InteractiveAI following the instructions provided in the [README](https://github.com/ainetus/InteractiveAI/blob/main/README.md) of the interface: 
In short, required steps are:
`export VITE_ATM_SIMU=http://localhost:6100`
```bash
cd config/dev/cab-standalone
./docker-compose.sh
```
Configuring Keycloak (credentials -> admin:admin): Realm Settings → Frontend URL → set to `http://localhost:3200`
Restart the frontend:
```bash
docker restart frontend
```
Load resources:
```bash
cd ../../../resources
./loadTestConf.sh
```
Log in:
Go to `http://localhost:3200`, log in as `atm_user` / `test`.

2. Run BlueSky in detached mode using the connector 'ai4realnet_rl_batch_bridge.py'.

The following dependencies are required:

```bash
cd bluesky
pip install -e .
pip install flask flask-cors requests
pip install stable_baselines3
```

Running the connector:
```bash 
python InteractiveAI_connectors/ai4realnet_rl_batch_bridge.py --port 6100 --cab-url http://localhost:3200/ --cab-user atm_user --cab-password test --plugin None --scenario ai4realnet_deploy_RL_batch/ai4realnet_deploy_RL_single_scn.scn
```

Other (optional) arguments are available. An example of such features is the speed of the simulation.



The connector also supports display and interaction with batch scenarios in InteractiveAI, for experiments with repeated conditions. This can be achieved through the plugin `ai4realnet_deploy_RL_batch.py`.
For this case, run the connector as follows:
```bash
python InteractiveAI_connectors/ai4realnet_rl_batch_bridge.py --port 6100 --cab-url http://localhost:3200/ --cab-user atm_user --cab-password test
```

## Project Structure

The main BlueSky files used by the connector are:
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
    ├── ai4realnet_deploy_RL_single_scn.scn or your own customised scenario
    ├── config_screen.scn
    ├── sector.scn
    └── generated_scenarios/
```
The above folder `generated_scenarios` contains the scenarios generated while running the RL deployment, saved through saveic.

The scenario used in the script above (ai4realnet_deploy_RL_single_scn.scn) for running the connector uses a synthetic scenario generator, as in the format below.
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
```
The following parameters can be modified for testing: `<N_AC>, <N_OBSTACLES>, <ENVIRONMENT>, <ALGORITHM>, <N_SCN>`.
Alternatively, the historical data can be used in by replacing the first six lines in the example scenario above with the following formatted data:
```
00:00:00.00>POLY SECTOR <lat_point1> <lon_point1> <lat_point2> <lon_point2> ...  <lat_pointN> <lon_pointN>
00:00:00.00>TAXI OFF 1000
00:00:00.00>POLY RESTRICTED_AREA_1, <lat_point1> <lon_point1> <lat_point2> <lon_point2> ...  <lat_pointN> <lon_pointN>
.
.
.
00:00:00.00>CRE AC1, <type>, <lat_point_orig>, <lon_point_orig>, <heading>, <speed>, <altitude>
00:00:00.00>DEST AC1 <lat_point_dest>, <lon_point_dest>
.
.
.
```


## Authors
- [Giulia Leto](https://github.com/giulialeto)

### Contributors

This project exists thanks to all the people who contribute to [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky), to [BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym), to [InteractiveAI](https://github.com/ainetus/InteractiveAI) and to the [MORL-DOL library](https://github.com/AI4REALNET/Grid2Op_MORL)

<a href="https://github.com/TUDelft-CNS-ATM/bluesky/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TUDelft-CNS-ATM/bluesky" />
</a>

<a href="https://github.com/TUDelft-CNS-ATM/bluesky-gym/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TUDelft-CNS-ATM/bluesky-gym" />
</a>

<a href="https://github.com/IRT-SystemX/InteractiveAI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=IRT-SystemX/InteractiveAI" />
</a>

<!-- TODO: Fill in the contributions of MORL-DOL library -->

## License
MIT License

Copyright (c) 2026 TU Delft
