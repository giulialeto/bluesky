# ATM BlueSky Assignment
As part of the Air Traffic Management (ATM) lecture at TU Delft, 2025.

... TODO add some description

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)

## Background

The background for this plugin is bifold:
1. Flight sectors are assigned to Air traffic controllers (ATCOs). Depending on the busyness of individual sectors, different (sub)sectorization might be applied.
2. Climate sensitive regions (CSRs) are areas where we expect a large climate impact of aircraft due to large non-CO2 effects.

> The goal of this plugin is to analyze the impact of CSR avoidance on the workload of ATCOs and the thus induced frequency of sector changes.

## Install

Instructions to install BlueSky can be found in the [BlueSky Wiki](https://github.com/TUDelft-CNS-ATM/bluesky/wiki/Installation).

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

Run BlueSky using `python BlueSky.py` and activate the plugin with `PLUGIN ASSIGNMENT`.

To load the historic air traffic data, request the scenario file from the authors and load it using `PCALL scenario_filename.SCN`.

# Personal BlueSky Manual
Commands, code snippets, and simulator's behavior that we consider important.

## General
- Aircraft only disappear from the simulation if a) the runway is given in the aircraft definition or b) taxiing is disabled, i.e. `TAXI OFF`.

## Scenario files
- Historic trajectories can be mimicked using the `RTA` command:

```
# Create a flight KL123, using the B738 aircraft, from Paris to Lanzarote.
00:00:00.00>CRE KL123 B738 LFPO 0 0 400
00:00:00.00>ORIG KL123 LFPO
00:00:00.00>DEST KL123 GCRR

# Create a waypoint wp1 which should be reached at 11:42:00.
00:00:00.00>DEFWPT wp1 48.72333,2.37945, FIX
00:00:00.00>ADDWPT KL123 wp1 0
00:00:00.00>RTA KL123 wp1 11:42:00.00
```

- New waypoints can be inserted into the trajectory, e.g. after a specific waypoint:
```
# Add three waypoints to flight KL123.
00:00:00.00>ADDWPT KL123 wp1
00:00:00.00>ADDWPT KL123 wp2
00:00:00.00>ADDWPT KL123 wp3

# Add waypoint wp23 in between wp2 and wp3
00:00:00.00>DEFWPT wp23 lat,lon FIX
00:00:00.00>ADDWPT KL123 wp23 0 0 wp2
```

## Python code
- get the trajectory of a flight

```python
from bluesky import traf
ac_idx = traf.id2idx("KL123")
if ac_idx != -1:
    trajectory = traf.ap.route[ac_idx]
```

- add a new stack command with a boolean parameter that can be called via `DEMO1`
```python
from bluesky import stack
@stack.command(name="DEMO1")
def demo_1(self, enable: "bool"):
    print(f"Feature enabled: {enable}")
```

```
# Call it using any of the three options
00:00:00.00>DEMO1 ON/OFF
00:00:00.00>DEMO1 1/0
00:00:00.00>DEMO1 True/False
```


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
- Sector information from ...
