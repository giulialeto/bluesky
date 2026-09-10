#!/usr/bin/env python
"""
ai4realnet_rl_batch_bridge.py

HTTP bridge between BlueSky's AI4REALNET ATM
use case 2 plugin and InteractiveAI.

This bridge boots BlueSky in detached mode, loads the ai4realnet_deploy_RL_batch plugin & scenario:

    PLUGIN <plugin>            (default: deployRL_batch, i.e.
                                 bluesky/plugins/ai4realnet_deploy_RL_batch.py)
    DETACHED_BATCH <scenario>  (default: scenario/ai4realnet_deploy_RL_batch/
                                 ai4realnet_deploy_RL_batch.scn)

and then streams the scenario state to InteractiveAI.


Architecture
------------
    BlueSky (embedded, background threads)
        --push context (aircraft state)--> InteractiveAI context-service
        --push events  (log/lifecycle)-->  InteractiveAI event-service
        <--receive chosen action--         InteractiveAI frontend (POSTs
                                            to THIS bridge's /update-flight-plan)

1. Aircraft context (continuous). Every PUSH_INTERVAL_S seconds, the state
   of every aircraft currently in the simulation (id, speed, lat, lon) is
   POSTed to context-service. Matches MetadataSchemaATM in
   backend/context-service/resources/ATM/schemas.py.

2. BlueSky's ECHO messages are forwarded to InteractiveAI as an event.

3. Aircraft/weather/volcanic lifecycle (polled, event-driven). The RL agent
   deletes an aircraft once it reaches its destination (see `update()` in
   ai4realnet_deploy_RL_batch.py), and the disturbance_generator plugin
   spawns/removes WEATHER_CELL and VOLCANIC_CELL shapes. PUSH_INTERVAL_S poll
   checks traf.id and the areafilter shape set against the previous poll and
   emits AIRCRAFT_SPAWNED / AIRCRAFT_REMOVED / WEATHER_CELL_* /
   VOLCANIC_CELL_* events for whatever changed.

Usage
-----
    cd bluesky
    pip install -e .  # install BlueSky in dev mode
    pip install flask flask-cors requests
    pip install stable_baselines3

    python ai4realnet_rl_batch_bridge.py \\
        --port 5100 \\
        --cab-url http://localhost:3200/ \\
        --cab-user atm_user --cab-password test

Then point InteractiveAI's frontend build at this bridge:
    export VITE_ATM_SIMU=http://localhost:5100

"""

import argparse
import base64
import queue
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

import bluesky as bs
from bluesky import stack

app = Flask(__name__)
CORS(app)  # dev-only: wide open.

LOOP_SLEEP = 0.1
PUSH_INTERVAL_S = 5

_sim_lock = threading.Lock()
_sim_running = False
_scenario_started = False

# Everything destined for InteractiveAI's event-service is queued here so
# that neither the sim thread nor the poll thread ever blocks on an HTTP call.
EVENT_QUEUE = queue.Queue()

# Snapshot of the world as of the previous poll, used to diff for lifecycle
_prev_aircraft_ids = set()
_prev_disturbance_shapes = set()

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
    global _sim_running
    _sim_running = True
    while _sim_running:
        with _sim_lock:
            bs.sim.step()
        time.sleep(LOOP_SLEEP)

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
    stack.stack(f"PLUGIN {CONFIG['plugin']}")
    stack.stack(f"DETACHED_BATCH {CONFIG['scenario']}")

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
        if name == "WEATHER_CELL":
            kind = "WEATHER"
        elif name == "VOLCANIC_CELL":
            kind = "VOLCANIC"
        elif name == CONFIG.get("sector_name"):
            kind = "SECTOR"
        else:
            kind = "OBSTACLE"
        shapes.append({"name": name, "kind": kind, "coordinates": points})
    return shapes


def _aircraft_in_los():
    """Set of acids currently in a loss of separation per BlueSky's own conflict
    detection module (traf.cd.lospairs)."""
    los_acids = set()
    for a, b in getattr(bs.traf.cd, "lospairs", []):
        los_acids.add(a)
        los_acids.add(b)
    return los_acids


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
        "date": datetime.now(timezone.utc).isoformat(),
        "data": {"airplanes": airplanes, "shapes": build_shapes_payload()},
    }


def push_context():
    url = CONFIG["cab_url"] + "cabcontext/api/v1/contexts"
    _post_with_relogin(url, build_context_payload())


def push_event(*, id_plane, system, event_type, title, description,
               criticality="MEDIUM", is_active=True, duration_minutes=5):
    """Pushes an event matching MetadataSchemaATM
    (backend/event-service/resources/ATM/schemas.py: event_type, system,
    id_plane required)."""
    now = datetime.now(timezone.utc)
    end = now.timestamp() + duration_minutes * 60
    payload = {
        "criticality": criticality,
        "title": title[:255],
        "description": description[:255],
        "start_date": now.isoformat(),
        "end_date": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
        "data": {"event_type": event_type, "system": system, "id_plane": id_plane},
        "use_case": "ATM",
        "is_active": is_active,
    }
    url = CONFIG["cab_url"] + "cab_event/api/v1/events"
    _post_with_relogin(url, payload)


def _push_echo_event(text, flags):
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
    diffs the world for lifecycle events (sources #1 and #3)."""
    global _prev_aircraft_ids, _prev_disturbance_shapes
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

            for acid in current_ids - _prev_aircraft_ids:
                EVENT_QUEUE.put(("aircraft", acid, "AIRCRAFT_SPAWNED",
                                  f"Aircraft {acid} entered the sector",
                                  f"Aircraft {acid} was spawned by the RL batch scenario.",
                                  "ROUTINE"))
            for acid in _prev_aircraft_ids - current_ids:
                EVENT_QUEUE.put(("aircraft", acid, "AIRCRAFT_REMOVED",
                                  f"Aircraft {acid} left the sector",
                                  f"Aircraft {acid} was removed (destination reached or batch reset).",
                                  "ROUTINE"))
            _prev_aircraft_ids = current_ids

            for shape_name in current_disturbances - _prev_disturbance_shapes:
                EVENT_QUEUE.put(("disturbance", shape_name, f"{shape_name}_SPAWNED",
                                  f"{shape_name.replace('_', ' ').title()} appeared",
                                  f"{shape_name} disturbance activated.",
                                  "MEDIUM"))
            for shape_name in _prev_disturbance_shapes - current_disturbances:
                EVENT_QUEUE.put(("disturbance", shape_name, f"{shape_name}_CLEARED",
                                  f"{shape_name.replace('_', ' ').title()} cleared",
                                  f"{shape_name} disturbance is no longer active.",
                                  "ROUTINE"))
            _prev_disturbance_shapes = current_disturbances
           
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
                _, id_plane, event_type, title, description, criticality = item
                system = "BLUESKY_RL_BATCH" if kind == "aircraft" else "ENVIRONMENT"
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
    return jsonify({
        "status": "ok",
        "sim_running": _sim_running,
        "scenario_started": _scenario_started,
        "logged_in": _access_token is not None,
        "plugin": CONFIG.get("plugin"),
        "scenario": CONFIG.get("scenario"),
        "n_aircraft": n_aircraft,
        "pending_events": EVENT_QUEUE.qsize(),
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
                              "ai4realnet_deploy_RL_batch.py's init_plugin())")
    parser.add_argument("--scenario", default="ai4realnet_deploy_RL_batch/ai4realnet_deploy_RL_batch.scn",
                         help="Path passed to DETACHED_BATCH, relative to settings.cfg's "
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
    args = parser.parse_args()

    PUSH_INTERVAL_S = args.push_interval

    CONFIG.update({
        "cab_url": args.cab_url if args.cab_url.endswith("/") else args.cab_url + "/",
        "cab_user": args.cab_user,
        "cab_password": args.cab_password,
        "plugin": args.plugin,
        "scenario": args.scenario,
        "acid": args.acid,
        "sector_name": args.sector_name,
    })

    init_bluesky()
    _scenario_started = True
    threading.Thread(target=sim_loop, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()
    threading.Thread(target=event_worker, daemon=True).start()
    threading.Thread(target=_cd_activation, daemon=True).start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
