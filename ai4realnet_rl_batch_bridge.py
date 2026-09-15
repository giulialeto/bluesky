#!/usr/bin/env python
"""
ai4realnet_rl_batch_bridge.py

HTTP bridge between BlueSky's AI4REALNET ATM use case 2 plugin and InteractiveAI.

This bridge boots BlueSky in detached mode, loads the ai4realnet_deploy_RL_batch plugin & scenario:

    PLUGIN <plugin>            (default: deployRL_batch, i.e.
                                 bluesky/plugins/ai4realnet_deploy_RL_batch.py)
    DETACHED_BATCH <scenario>  (default: scenario/ai4realnet_deploy_RL_batch/
                                 ai4realnet_deploy_RL_batch.scn)

and then streams the scenario state to InteractiveAI.

Architecture
------------
    BlueSky
        push context (aircraft state)--> InteractiveAI context-service
        push events  (log/lifecycle)-->  InteractiveAI event-service
        <--receive chosen actions:  InteractiveAI frontend (POSTs to bridge's /update-flight-plan)

1. Aircraft context (continuous). Every PUSH_INTERVAL_S seconds, the state
   of every aircraft currently in the simulation (id, speed, lat, lon) is
   POSTed to context-service. Matches MetadataSchemaATM in
   backend/context-service/resources/ATM/schemas.py.

2. BlueSky's ECHO messages are forwarded to InteractiveAI as an event.

3. Aircraft/weather/volcanic lifecycle (polled, event-driven). The RL agent
   deletes an aircraft once it reaches its destination, and the disturbance_generator plugin
   spawns/removes WEATHER_CELL and VOLCANIC_CELL shapes. PUSH_INTERVAL_S poll
   checks traf.id and the areafilter shape set against the previous poll and
   emits AIRCRAFT_SPAWNED / WEATHER_CELL_* / VOLCANIC_CELL_* events for
   whatever changed.

4. Aircraft area incursions (polled). Every PUSH_INTERVAL_S poll also checks
   each in-sector aircraft against restricted areas (a
   WEATHER_CELL/VOLCANIC_CELL perturbation, or any restricted area the
   scenario defines) using BlueSky's own
   areafilter.checkInside(), and emits an AIRCRAFT_IN_<shape name> event when
   an aircraft enters or leaves one.

Usage
-----
    cd bluesky
    pip install -e .  # install BlueSky in dev mode
    pip install flask flask-cors requests
    pip install stable_baselines3

    python ai4realnet_rl_batch_bridge.py \\
        --port 6100 \\
        --cab-url http://localhost:3200/ \\
        --cab-user atm_user --cab-password test

To run a plain scenario without using deployRL_batch's custom initialize_scenario
    python ai4realnet_rl_batch_bridge.py \\
        --plugin None \\
        --scenario scenario.scn

Then point InteractiveAI's frontend build at this bridge:
    export VITE_ATM_SIMU=http://localhost:6100

"""

import argparse
import base64
import queue
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

import bluesky as bs
from bluesky import stack
# Only to fetch the duration of the perturbation - the perturbations are generated from the bluesky scenario: `PLUGIN disturbance_generator`
import bluesky.plugins.ai4realnet_perturbations as perturbations_plugin

app = Flask(__name__)
CORS(app)

PUSH_INTERVAL_S = 5

_sim_lock = threading.Lock()
_sim_running = False
_scenario_started = False

# Everything destined for InteractiveAI's event-service is queued here so
# that neither the sim thread nor the poll thread ever blocks on an HTTP call.
EVENT_QUEUE = queue.Queue()

# Snapshot of the world as of the previous poll
_prev_aircraft_ids = set()
_prev_disturbance_shapes = set()
_prev_los_pairs = set()
_prev_area_incursions = set()  # (acid, shape_name) pairs currently inside a non-sector area shape

# Event types whose lifecycle has a end. The "start" push leaves endDate empty if the end time is unknown.
# The end of the event triggers an update to the same event_type with the endDate set.
_OPEN_ENDED_EVENT_TYPES = {"WEATHER_CELL", "VOLCANIC_CELL", "AIRCRAFT_LOS", "AIRCRAFT_SPAWNED"}

# Labels for each event recording the startDate, so the update at end of life does not overwrite it.
_open_condition_start = {}

# opfab-client's public client id/secret, base64'd -- same constant the
# PowerGrid example uses (usecases_examples/PowerGrid/app/models/Communicate.py)
_OPFAB_CLIENT_BASIC_AUTH = "Basic " + base64.b64encode(b"opfab-client:opfab-keycloak-secret").decode()

CONFIG = {}  # populated in main() from CLI args
_access_token = None

# BlueSky's stack echo flags (bluesky/__init__.py): BS_OK=0, BS_ARGERR=1,
# BS_FUNERR=2, BS_CMDERR=4. Tune if you want scenario-progress ECHOs and stack warnings to
# show up with different InteractiveAI criticality.
_ECHO_CRITICALITY = {getattr(bs, "BS_OK", 0): "ROUTINE"}


# --------------------------------------------------------------------------
# BlueSky sim loop
# --------------------------------------------------------------------------

def sim_loop():
    """Drives BlueSky forward using bs.sim.update()
    bs.sim.dtmult is used to control the speed of the simulation, set by adding the argument --sim_speed to main. 
    Overrides the default DTMULT set within the original plugins, to allow human in the loop studies.
    """
    global _sim_running
    _sim_running = True
    while _sim_running:
        with _sim_lock: # bs.sim.update() must be called with the sim lock held, since it reads/writes bs.sim.* state.
            target_speed = CONFIG.get("sim_speed", 1.0)
            if bs.sim.dtmult != target_speed:
                bs.sim.set_dtmult(target_speed)
            bs.sim.update()

# Remove _SUPPRESSED_ECHO_PREFIXES and _capture_net_send after finalising the development. The user does not need to see these messages. 
_SUPPRESSED_ECHO_PREFIXES = (
    # Remove ECHO messages that are not useful to the operator from the event stream.
    "Selected StateBased as CD method.",
)


def _capture_net_send(topic, data='', to_group=b''):
    """Replaces bluesky.network.detached.Node.send so that ECHO
        messages are captured instead. This runs on the sim thread,
        inside _sim_lock."""
    try:
        topic_str = topic.decode() if isinstance(topic, bytes) else topic
        if topic_str == "ECHO" and isinstance(data, dict):
            text = data.get("text", "")
            if text.strip().startswith(_SUPPRESSED_ECHO_PREFIXES):
                return
            EVENT_QUEUE.put(("echo", text, data.get("flags", 0)))
    except Exception as exc:
        print(f"[bridge] failed to capture net.send({topic!r}): {exc}")


def init_bluesky():
    bs.init(mode="sim", detached=True)
    bs.net.send = _capture_net_send  # start capturing BlueSky's log stream immediately
    if CONFIG["plugin"] is not None:
        # This is meant to load the 'detatched_batch' plugin
        stack.stack(f"PLUGIN {CONFIG['plugin']}")
        stack.stack(f"DETACHED_BATCH {CONFIG['scenario']}")
    else:
        # No plugin: load the scenario the normal BlueSky way.
        stack.stack(f"IC {CONFIG['scenario']}")

def _cd_activation():
    """Keeps conflict detection active (detached batch logic de-activates it at every reset).
        This is used to show the protected zones in red in the front end, if separation is lost."""
    while True:
        time.sleep(2)
        with _sim_lock:
            resolved = bs.traf.cd.__dict__.get("_refobj")
            if type(resolved).__name__ != "StateBased":
                stack.stack("CDMETHOD ON")

# --------------------------------------------------------------------------
# InteractiveAI (CAB) client -- login + push context/events
# --------------------------------------------------------------------------

def cab_login():
    """Logs into InteractiveAI's Keycloak and stores a bearer token."""
    global _access_token
    url = CONFIG["cab_url"] + "auth/token"
    payload = (
        f"username={CONFIG['cab_user']}&password={CONFIG['cab_password']}"
        "&grant_type=password&clientId=opfab-client"
    )
    headers = {
        "Authorization": _OPFAB_CLIENT_BASIC_AUTH,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = requests.post(url, headers=headers, data=payload, timeout=15)
    response.raise_for_status()
    _access_token = response.json().get("access_token")
    return _access_token is not None


def cab_auth_headers():
    return {
        "Authorization": f"Bearer {_access_token}",
        "Content-Type": "application/json",
    }


def _post_with_relogin(url, payload):
    response = requests.post(url, headers=cab_auth_headers(), json=payload, timeout=15)
    if response.status_code == 401:
        cab_login()
        response = requests.post(url, headers=cab_auth_headers(), json=payload, timeout=15)
    response.raise_for_status()
    return response


def _shape_kind(name):
    """Classifies a named BlueSky area shape (areafilter.basic_shapes key):
    the two perturbation shapes the disturbance_generator plugin spawns, the
    scenario's own sector polygon (--sector-name), or an obstacle."""
    if name == "WEATHER_CELL":
        return "WEATHER"
    elif name == "VOLCANIC_CELL":
        return "VOLCANIC"
    elif name == CONFIG.get("sector_name"):
        return "SECTOR"
    else:
        return "OBSTACLE"


def build_shapes_payload():
    """Builds the sector/weather/volcanic shape list matching
    ShapeMetadataSchemaATM (backend/context-service/resources/ATM/schemas.py).
    Reads BlueSky's named area shapes (areafilter.basic_shapes), which
    includes the sector polygon defined in the scenario plus any
    WEATHER_CELL/VOLCANIC_CELL disturbances the disturbance_generator plugin
    has spawned."""
    shapes = []
    basic_shapes = getattr(bs.tools.areafilter, "basic_shapes", {})
    for name, shape in basic_shapes.items():
        coordinates = list(shape.coordinates)
        # coordinates is a flat [lat0, lon0, lat1, lon1, ...] list
        points = [[float(lat), float(lon)]
                  for lat, lon in zip(coordinates[::2], coordinates[1::2])]
        if not points:
            continue
        shapes.append({"name": name, "kind": _shape_kind(name), "coordinates": points})
    return shapes


def _aircraft_area_incursions():
    """Set of (acid, shape_name) pairs for every aircraft currently inside a
    non-sector named area shape -- a WEATHER_CELL/VOLCANIC_CELL perturbation,
    or a restricted area -- using BlueSky's own
    areafilter.checkInside()"""
    incursions = set()
    ids = list(bs.traf.id)
    if not ids:
        return incursions
    basic_shapes = getattr(bs.tools.areafilter, "basic_shapes", {})
    for name in basic_shapes:
        if _shape_kind(name) == "SECTOR":
            continue  # the sector is the whole controlled airspace
        inside = bs.tools.areafilter.checkInside(name, bs.traf.lat, bs.traf.lon, bs.traf.alt)
        for acid, is_inside in zip(ids, inside):
            if is_inside:
                incursions.add((acid, name))
    return incursions


def _aircraft_in_los():
    """Set of acids currently in a loss of separation per BlueSky's own conflict
    detection module (traf.cd.lospairs)."""
    los_acids = set()
    for a, b in getattr(bs.traf.cd, "lospairs", []):
        los_acids.add(a)
        los_acids.add(b)
    return los_acids


def _unique_los_pairs():
    """Places each current loss of separation as a (acid1, acid2) tuple"""
    return {tuple(sorted(pair)) for pair in getattr(bs.traf.cd, "lospairs_unique", [])}


def sim_now():
    """Gives the simulation's current clock time as a UTC-aware datetime"""
    with _sim_lock:
        return bs.sim.utc.replace(tzinfo=timezone.utc)


def _disturbance_end_date(shape_name):
    """Fetches the end date of WEATHER_CELL or VOLCANIC_CELL disturbances, which is available by construction. Returns None if that state isn't available"""
    gen = perturbations_plugin.perturbation_generator
    if gen is None:
        return None
    if shape_name == "WEATHER_CELL" and getattr(gen, "weather_active", False):
        start_simt, lifetime = gen.weather_disturbance_start, gen.weather_cell_lifetime
    elif shape_name == "VOLCANIC_CELL" and getattr(gen, "volcanic_active", False):
        start_simt, lifetime = gen.volcanic_disturbance_start, gen.volcanic_cell_lifetime
    else:
        return None
    remaining_s = max(0.0, (start_simt + lifetime) - bs.sim.simt)
    return bs.sim.utc.replace(tzinfo=timezone.utc) + timedelta(seconds=remaining_s)


def build_context_payload():
    """Builds the ATM context payload matching MetadataSchemaATM
    (backend/context-service/resources/ATM/schemas.py)."""
    los_acids = _aircraft_in_los()
    airplanes = []
    for i in range(bs.traf.ntraf):
        airplanes.append({
            "id_plane": bs.traf.id[i],
            "Current_airspeed": float(bs.traf.gs[i]) * 1.94384,  # m/s -> knots
            "Latitude": float(bs.traf.lat[i]),
            "Longitude": float(bs.traf.lon[i]),
            # True heading, degrees clockwise from north (0-360) -- matches
            # both compass convention and CSS's rotate() direction, so the
            # frontend can point aircraft icons the right way. Requires the
            # matching `heading` field on PlaneMetadataSchemaATM in
            # backend/context-service/resources/ATM/schemas.py.
            "heading": float(bs.traf.hdg[i]),
            # True if another aircraft is currently inside this one's protected zone
            "in_los": bs.traf.id[i] in los_acids,
        })
    return {
        "use_case": "ATM",
        # Simulated time
        "date": bs.sim.utc.replace(tzinfo=timezone.utc).isoformat(),
        "data": {"airplanes": airplanes, "shapes": build_shapes_payload()},
    }


def push_context():
    url = CONFIG["cab_url"] + "cabcontext/api/v1/contexts"
    _post_with_relogin(url, build_context_payload())


def push_event(*, id_plane, system, event_type, title, description,
               criticality="MEDIUM", is_active=True, duration_minutes=5,
               start_date=None, end_date=None):
    """Pushes an event matching MetadataSchemaATM
    (backend/event-service/resources/ATM/schemas.py: event_type, system,
    id_plane required).

    start_date defaults to current sim time, unless an update to an existing card is being pushed, for which the original start_date is preserved.

    The endDate is set in sim time when available. Once endDate passes, the card is automatically removed from the active alerts. 
    Pass duration_minutes=None (and no end_date) to leave the card open-ended until a later update sets the endDate."""
    now = sim_now()
    if end_date is not None:
        end_date_iso = end_date.isoformat()
    elif duration_minutes is not None:
        end = now.timestamp() + duration_minutes * 60
        end_date_iso = datetime.fromtimestamp(end, tz=timezone.utc).isoformat()
    else:
        end_date_iso = None
    payload = {
        "criticality": criticality,
        "title": title[:255],
        "description": description[:255],
        "start_date": (start_date or now).isoformat(),
        "end_date": end_date_iso,
        "data": {"event_type": event_type, "system": system, "id_plane": id_plane},
        "use_case": "ATM",
        "is_active": is_active,
    }
    url = CONFIG["cab_url"] + "cab_event/api/v1/events"
    _post_with_relogin(url, payload)


def _push_echo_event(text, flags):
    ''' Pushes a BlueSky ECHO message to InteractiveAI's event-service. The first line of the ECHO is used as the title, and the rest as the description.
    Events from bluesky's ECHO are tagged with low priotity
    Can be deleted once the development is finalised.'''
    if not text:
        return
    criticality = _ECHO_CRITICALITY.get(flags, "LOW")
    title = text.strip().splitlines()[0] if text.strip() else "BlueSky log"
    push_event(
        id_plane="SYSTEM",  # not tied to one aircraft -- see MetadataSchemaATM (id_plane required)
        system="BLUESKY",
        event_type="SIM_LOG",
        title=title,
        description=text,
        criticality=criticality,
    )


# --------------------------------------------------------------------------
# Background threads: InteractiveAI push loop + event queue worker
# --------------------------------------------------------------------------

def push_loop():
    """Background thread: logs in, then periodically pushes context and
    diffs the world for lifecycle events."""
    global _prev_aircraft_ids, _prev_disturbance_shapes, _prev_los_pairs, _prev_area_incursions
    while not cab_login():
        time.sleep(5)
    while _sim_running:
        try:
            with _sim_lock:
                push_context()
                current_ids = set(bs.traf.id)
                shapes = getattr(bs.tools.areafilter, "basic_shapes", {})
                current_disturbances = {
                    name for name in shapes
                    if name in ("WEATHER_CELL", "VOLCANIC_CELL")
                }
                # check for aircraft pairs in a loss of separation
                current_los_pairs = _unique_los_pairs()
                # check for aircraft inside a perturbation or restricted area
                current_area_incursions = _aircraft_area_incursions()
                new_disturbances = current_disturbances - _prev_disturbance_shapes
                # For weather/volcanic disturbances, fetches the end date directly from the plugin's construction of the disturbance,
                disturbance_end_dates = {name: _disturbance_end_date(name) for name in new_disturbances}

            # Alert when an aircraft enters the sector or leaves it. 
            for acid in current_ids - _prev_aircraft_ids:
                EVENT_QUEUE.put(("aircraft", acid, "AIRCRAFT_SPAWNED",
                                  f"Aircraft {acid} entered the sector",
                                  " ",
                                  "ROUTINE", False, None))
            for acid in _prev_aircraft_ids - current_ids:
                EVENT_QUEUE.put(("aircraft", acid, "AIRCRAFT_SPAWNED",
                                  f"Aircraft {acid} left the sector",
                                  " ",
                                  # Card auto-clears 2 minutes after the aircraft actually
                                  # left, rather than the instant this push is processed.
                                  "ROUTINE", True, sim_now() + timedelta(minutes=2)))
            _prev_aircraft_ids = current_ids

            # The 2nd to last element of the tuple tells event_worker whether this is the
            # real end of the condition (True) or its start/an ongoing push
            # (False); the last is the disturbance's scheduled end (None at the end of the event, as the 2nd to last element already specifies True).
            for shape_name in new_disturbances:
                EVENT_QUEUE.put(("disturbance", shape_name, shape_name,
                                  f"{shape_name.replace('_', ' ').title()} appeared",
                                  " ",
                                  "MEDIUM", False, disturbance_end_dates[shape_name]))
            for shape_name in _prev_disturbance_shapes - current_disturbances:
                EVENT_QUEUE.put(("disturbance", shape_name, shape_name,
                                  f"{shape_name.replace('_', ' ').title()} cleared",
                                  " ",
                                  "ROUTINE", True, None))
            _prev_disturbance_shapes = current_disturbances

            # Alert fired when two aircraft are in a loss of separation or when the loss of separation is resolved. EndDate for these events is not known in advance.
            for acid1, acid2 in current_los_pairs - _prev_los_pairs:
                EVENT_QUEUE.put(("aircraft", acid1, "AIRCRAFT_LOS",
                                  f"Loss of separation: {acid1} - {acid2}",
                                  " ",
                                  "HIGH", False, None))
            for acid1, acid2 in _prev_los_pairs - current_los_pairs:
                EVENT_QUEUE.put(("aircraft", acid1, "AIRCRAFT_LOS",
                                  f"Loss of separation resolved: {acid1} - {acid2}",
                                  " ",
                                  "ROUTINE", True, None))
            _prev_los_pairs = current_los_pairs

            # Alert fired when an aircraft enters or leaves a perturbation (WEATHER_CELL/VOLCANIC_CELL) or a restricted area.
            for acid, shape_name in current_area_incursions - _prev_area_incursions:
                kind = _shape_kind(shape_name)
                label = shape_name.replace("_", " ").title()
                if kind in ("WEATHER", "VOLCANIC"):
                    title, criticality = f"Aircraft {acid} entered {label}", "MEDIUM"
                else:
                    title, criticality = f"Aircraft {acid} entered {label}", "HIGH"
                EVENT_QUEUE.put(("aircraft", acid, f"AIRCRAFT_IN_{shape_name}",
                                  title, " ", criticality, False, None))
            for acid, shape_name in _prev_area_incursions - current_area_incursions:
                label = shape_name.replace("_", " ").title()
                EVENT_QUEUE.put(("aircraft", acid, f"AIRCRAFT_IN_{shape_name}",
                                  f"Aircraft {acid} left {label}",
                                  " ",
                                  "ROUTINE", True, None))
            _prev_area_incursions = current_area_incursions

        except Exception as exc:  # simulator must keep running even if CAB is unreachable
            print(f"[push_loop] failed to push context: {exc}")
        time.sleep(PUSH_INTERVAL_S)


def event_worker():
    """Drains EVENT_QUEUE and pushes each item to InteractiveAI's
    event-service. Runs off the sim thread so a slow/unreachable CAB never
    stalls the simulation."""
    while not cab_login():
        time.sleep(5)
    while True:
        item = EVENT_QUEUE.get()
        try:
            kind = item[0]
            if kind == "echo":
                _, text, flags = item
                _push_echo_event(text, flags)
            elif kind in ("aircraft", "disturbance"):
                _, id_plane, event_type, title, description, criticality, resolved, scheduled_end = item
                system = "BLUESKY_RL_BATCH" if kind == "aircraft" else "ENVIRONMENT"
                if event_type in _OPEN_ENDED_EVENT_TYPES or event_type.startswith("AIRCRAFT_IN_"):
                    # Same id_plane + event_type on both the "start" and
                    # "end" push, so InteractiveAI's event-service treats
                    # the second as an update to the SAME card
                    key = (id_plane, event_type)
                    now = sim_now()
                    if resolved:
                        start = _open_condition_start.pop(key, now)
                        # Close it at the caller's scheduled_end if given, otherwise right now.
                        end = scheduled_end if scheduled_end is not None else now
                    else:
                        start = _open_condition_start.setdefault(key, now)
                        # Sets a scheduled end date if known in advance (WEATHER_CELL/VOLCANIC_CELL), None for AIRCRAFT_LOS,
                        # which leaves the card open until the "resolved" push supplies the end time.
                        end = scheduled_end
                    push_event(id_plane=id_plane, system=system, event_type=event_type,
                               title=title, description=description, criticality=criticality,
                               start_date=start, end_date=end, duration_minutes=None)
                else:
                    # Everything else uses a the default 5-minute auto-expiry for notifications.
                    push_event(id_plane=id_plane, system=system, event_type=event_type,
                               title=title, description=description, criticality=criticality)
        except Exception as exc:
            print(f"[event_worker] failed to push event {item!r}: {exc}")
        finally:
            EVENT_QUEUE.task_done()


# --------------------------------------------------------------------------
# HTTP endpoints
# --------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    with _sim_lock:
        n_aircraft = int(bs.traf.ntraf) if _sim_running else 0
        current_dtmult = float(bs.sim.dtmult) if _sim_running else None
    return jsonify({
        "status": "ok",
        "sim_running": _sim_running,
        "scenario_started": _scenario_started,
        "logged_in": _access_token is not None,
        "plugin": CONFIG.get("plugin"),
        "scenario": CONFIG.get("scenario"),
        "n_aircraft": n_aircraft,
        "pending_events": EVENT_QUEUE.qsize(),
        "sim_speed": CONFIG.get("sim_speed"),
        "dtmult": current_dtmult,
    })


@app.route("/state", methods=["GET"])
def state():
    """Debug helper -- InteractiveAI does not call this; use it to sanity-check
    what the context push is seeing."""
    with _sim_lock:
        payload = build_context_payload()
        payload["t"] = float(bs.sim.simt)
    return jsonify(payload)


@app.route("/command", methods=["POST"])
def command():
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get("command")
    if not cmd:
        return jsonify({"ok": False, "error": "missing 'command' field"}), 400
    # allows to change the simulation speed while running: POST {"command": "DTMULT <n>"}
    parts = cmd.strip().split()
    if len(parts) == 2 and parts[0].upper() == "DTMULT":
        try:
            CONFIG["sim_speed"] = float(parts[1])
        except ValueError:
            pass
    with _sim_lock:
        stack.stack(cmd)
    return jsonify({"ok": True, "command": cmd})


@app.route("/aircraft/<acid>", methods=["DELETE"])
def delete_aircraft(acid):
    with _sim_lock:
        stack.stack(f"DEL {acid}")
    return jsonify({"ok": True})


@app.route("/update-flight-plan", methods=["POST"])
def update_flight_plan():
    """Called by InteractiveAI's frontend when the operator applies a
    recommendation. Body matches Action<'ATM'> (frontend/src/entities/ATM/types.ts):

        {
          "airport_destination": {"apid": "EHAM", "latitude": 52.3, "longitude": 4.8, ...},
          "waypoints": [{"wpid": "...", "wplat": ..., "wplon": ..., "wpidx": 0}, ...]
        }

    No id_plane is included in this payload today, so it's applied to the
    single aircraft designated by --acid, if one was configured. In this
    RL-batch scenario every aircraft is currently under full autonomous
    control.
    """
    acid = CONFIG.get("acid")
    if not acid:
        return jsonify({
            "ok": False,
            "error": "no --acid configured for this bridge instance, and this "
                     "scenario's aircraft are RL-controlled; nothing to apply this to.",
        }), 400

    data = request.get_json(force=True, silent=True) or {}
    with _sim_lock:
        dest = data.get("airport_destination")
        if dest and dest.get("apid"):
            stack.stack(f"DEST {acid} {dest['apid']}")

        waypoints = sorted(data.get("waypoints", []), key=lambda wp: wp.get("wpidx", 0))
        for wp in waypoints:
            stack.stack(f"ADDWPT {acid} {wp['wplat']},{wp['wplon']}")

    return jsonify({"message": "ok", "acid": acid})


@app.route("/event/trigger", methods=["POST"])
def trigger_event():
    """Manual test hook: POST {"id_plane": "...", "system": "...", "event_type": "...",
    "title": "...", "description": "..."} to push a one-off event to InteractiveAI
    without waiting for real telemetry."""
    data = request.get_json(force=True, silent=True) or {}
    required = ["id_plane", "system", "event_type", "title", "description"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"ok": False, "error": f"missing fields: {missing}"}), 400
    try:
        push_event(**{k: data[k] for k in required},
                   criticality=data.get("criticality", "MEDIUM"),
                   is_active=data.get("is_active", True))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    return jsonify({"ok": True})


def main():
    global _scenario_started, PUSH_INTERVAL_S
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5100,
                         help="HTTP port this bridge serves on (set VITE_ATM_SIMU to this)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cab-url", default="http://localhost:3200/",
                         help="InteractiveAI frontend base URL, trailing slash required")
    parser.add_argument("--cab-user", default="atm_user")
    parser.add_argument("--cab-password", default="test")
    parser.add_argument("--plugin", default="deployRL_batch",
                         help="Plugin to load (matches plugin_name in "
                              "ai4realnet_deploy_RL_batch.py's init_plugin()). "
                              "Pass 'None' (case-insensitive) or an empty string to skip "
                              "loading a plugin and load --scenario with plain IC instead ")
    parser.add_argument("--scenario", default="ai4realnet_deploy_RL_batch/ai4realnet_deploy_RL_batch.scn",
                         help="Path passed to DETACHED_BATCH (or IC if --plugin is None), relative to settings.cfg's "
                              "scenario_path ('scenario/')")
    parser.add_argument("--push-interval", type=float, default=PUSH_INTERVAL_S,
                         help="Seconds between context pushes / lifecycle-diff polls")
    parser.add_argument("--acid", default=None,
                         help="Optional: aircraft ID /update-flight-plan applies to. "
                              "Not needed for pure monitoring.")
    parser.add_argument("--sector-name", default="LISBON_FIR",
                         help="Name of the sector area shape (as defined in the scenario "
                              "via e.g. `BOX`/`POLY` stack commands) to tag with kind=SECTOR "
                              "in the shapes payload. Any other named shape that isn't "
                              "WEATHER_CELL/VOLCANIC_CELL is tagged kind=OBSTACLE.")
    parser.add_argument("--sim-speed", type=float, default=1.0,
                         help="BlueSky simulation-speed multiplier (dtmult): 1.0 runs the "
                              "simulated clock at real-time, N runs it N times faster."
                              "Also adjustable while running via POST /command "
                              "{'command': 'DTMULT <n>'}.")
    args = parser.parse_args()

    PUSH_INTERVAL_S = args.push_interval

    # Accept 'None'/'none'/'' 
    plugin = args.plugin if args.plugin and args.plugin.strip().lower() != "none" else None

    CONFIG.update({
        "cab_url": args.cab_url if args.cab_url.endswith("/") else args.cab_url + "/",
        "cab_user": args.cab_user,
        "cab_password": args.cab_password,
        "plugin": plugin,
        "scenario": args.scenario,
        "acid": args.acid,
        "sector_name": args.sector_name,
        "sim_speed": args.sim_speed,
    })

    init_bluesky()
    _scenario_started = True
    threading.Thread(target=sim_loop, daemon=True).start()

    while not cab_login():
        time.sleep(5)

    threading.Thread(target=push_loop, daemon=True).start()
    threading.Thread(target=event_worker, daemon=True).start()
    threading.Thread(target=_cd_activation, daemon=True).start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
