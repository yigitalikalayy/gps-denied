# gps-denied-navigation

GPS‑denied navigation pipeline using optical flow, IMU gyro compensation, and range sensing. The system publishes MAVLink `OPTICAL_FLOW_RAD` and `DISTANCE_SENSOR` messages for PX4 EKF2 fusion.

This branch is configured for a companion-computer topology where `mavlink-router` owns the PX4 UART link and this application exchanges MAVLink with the router over localhost UDP.

## Repo Layout
```
├── docs/
├── src/
│   └── algorithms/
│       └── optical_flow/
│   ├── sensors/
│   └── mavlink_bridge/
├── analysis/
├── logs/
│   ├── simulation/
│   └── flights/
├── datasets/
│   └── bag_files/
├── scripts/
│   └── systemd/
└── config.json
```

## Quick Start (Real Hardware)
Algorithm is started via the shell wrapper:
```
./scripts/executor_mavlink_router_udp.sh config.json
```

Expected runtime topology:
```
optical_flow_app <-> UDP 127.0.0.1:14560 <-> mavlink-router <-> UART <-> PX4
QGroundControl <-> IP/UDP <-> mavlink-router
```

The default `config.json` now uses the `mavlink` section with `transport = udp`.
If you want to bypass `mavlink-router` and talk to PX4 directly again, switch `mavlink.transport` back to `serial`.

## Logging
Per‑run CSV logs are written to:
- `logs/flights/`

Logging is configured under `logging.flight_log` in `config.json`.

## Analysis
Use the scripts under `analysis/` to compute basic metrics:
```
python3 analysis/feature_stats.py logs/flights/
python3 analysis/altitude_analysis.py logs/flights/
python3 analysis/drift_analysis.py logs/flights/
```

## Docs
- `docs/architecture.md` — system overview
- `docs/optical_flow.md` — algorithm details
- `docs/experiment_notes.md` — experiment template and tips
- `scripts/systemd/mavlink-router-main.conf.sample` — sample router config for PX4 + QGC + optical flow app
