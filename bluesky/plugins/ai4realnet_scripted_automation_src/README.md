# AI4REALNET D1.2 Scripted automation (workload and re-routing) plugin for BlueSky,supporting use case ATM-UC2

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)

## Background

The background for this plugin is bifold:
1. Flight sectors are assigned to Air traffic controllers (ATCOs). Depending on the busyness of individual sectors, different (sub)sectorization might be applied.
2. Avoidance of sectors, for example military restricted areas or climate sensitive regions (CSRs, areas where we expect a large climate impact of aircraft due to large non-CO2 effects).

> The goal of this plugin is to analyze the impact of sector avoidance on the workload of ATCOs and the thus induced frequency of sector changes.

## Install

Instructions to install BlueSky can be found in the [BlueSky Wiki](https://github.com/TUDelft-CNS-ATM/bluesky/wiki/Installation).
Works with BlueSky versions up to Commit 849d76f

To use this plugin, it is also required to install Larp:

```sh
$ pip install -e git+https://github.com/jsmretschnig/Larp.git#egg=Larp
```

and pyproj:
```sh
$ pip install pyproj
```

Further dependencies are `numpy`, `pandas`, and `matplotlib`.

## Usage

Run BlueSky using `python BlueSky.py` and activate the plugin with `PLUGIN SCRIPTED_AUTOMATION`.

To load the historic air traffic data, request the scenario file from the authors and load it using `PCALL scenario_filename.SCN`.

### Flight sector functionality
- The workload evaluation function is active by default, with the assumption that 3 air traffic controllers (ATCOs) are available and that a maximum of 10 aircraft can be handled at the same time by 1 ATCO. Two options are available as stack commands. `AVAILABLE_ATCO` can be set from 1 to 6 (maximum number of available sector options). `ALLOWED_AIRCRAFT` can be used as a soft constraint to restrict the number of aircraft allowed in each seector.
- How the algorithm works: Every 15 minutes, the workload prediction algorithm is triggered, making a new sector opening plan based on the current secctor count and the estimation of the sector count for the future 15 minutes time span, based on the RTAs of the flights at the next waypoints of the flight plan. The algorithm favors opening as little sector a possible while the number of allowed sectors is below the maximum allowed, and to balance workload amonts the available controllers when that number is exceeded.
- The workload algorithm currently only supports the portuguese upper airspace, for which the most commonly used sectors have been imported. Extention to other airspaces is straightforward. 

### CSR functionality
- There are 3 boolean stack functions available: `CREATE_CSR` to generate random CSRs within the Portuguese airspace, `AVOID_CSR` to enable or disable the rerouting around CSR regions, and `PLOT_POTFIELD` to visualize the potential field and the graph network with the chosen route.
- How the algorithm works: The aircraft is navigating towards a certain waypoint (=origin). Whenever the 2nd upcoming waypoint (i.e. the waypoint after the one the aircraft is navigating to) lies within a CSR, the algorithm searches for the first waypoint outside the CSR again (=destination). It then uses these two waypoints as origin and destination for the route around the CSR, which is determined using quad trees, potential fields and the A* algorithm [(Rivera et al. 2024)](#references). Once a route was found, it replaces the waypoints within the CSR with the new ones. This is accomplished with the `ADDWPT` command.

### References
Rivera, Josue N., and Dengfeng Sun. "Multi-Scale Cell Decomposition for Path Planning using Restrictive Routing Potential Fields." arXiv preprint arXiv:2408.02786 (2024).

## Authors

- [Giulia Leto](https://github.com/giulialeto)
- [Jakob Smretschnig](https://github.com/jsmretschnig)

### Contributors

This project exists thanks to all the people who contribute to [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky).

<a href="https://github.com/TUDelft-CNS-ATM/bluesky/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TUDelft-CNS-ATM/bluesky" />
</a>

## License

[GNU](LICENSE) © TU Delft.
- Eurocontrol data from [https://www.eurocontrol.int/dashboard/rnd-data-archive](Aviation data for research)
- Sector information from [https://ais.nav.pt/online-eaip-en/](NAV Portugal)
