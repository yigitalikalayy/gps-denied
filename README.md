# gps-denied-navigation

GPS‑denied navigation pipeline using optical flow, IMU gyro compensation, and range sensing. The system publishes MAVLink `OPTICAL_FLOW_RAD` and `DISTANCE_SENSOR` messages for PX4 EKF2 fusion.

## Repo Layout
```
├── docs/
├── src/
│   ├── optical_flow/
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
└── config_sitl.json
```

## Quick Start (Simulation)
Algorithm is started via the shell wrapper:
```
./scripts/simulation.sh config_sitl.json
```

Optional (direct Python):
```
python3 scripts/run_optical_flow.py --config config_sitl.json
```

## Logging
Per‑run CSV logs are written to:
- Simulation: `logs/simulation/`
- Real flights: `logs/flights/`

Logging is configured under `logging.flight_log` in the config files.

## Analysis
Use the scripts under `analysis/` to compute basic metrics:
```
python3 analysis/feature_stats.py logs/simulation/
python3 analysis/altitude_analysis.py logs/simulation/
python3 analysis/drift_analysis.py logs/simulation/
```

## Docs
- `docs/architecture.md` — system overview
- `docs/optical_flow.md` — algorithm details
- `docs/experiment_notes.md` — experiment template and tips
