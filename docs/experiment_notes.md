# Experiment Notes

## Purpose
Track configuration changes, environment conditions, and observed behavior for each flight.

## Template
```
Date:
Environment: (Real)
Location:
Altitude Range:
Lighting / Texture:
Config File:
Changes Made:
Observations:
- Stability:
- Drift:
- Flow quality:
- EKF flags:
- Notes:

Next Actions:
```

## Suggested Logs to Capture
- `estimator_status_flags`
- `estimator_aid_src_optical_flow`
- `sensor_optical_flow`
- `vehicle_local_position`

## Common Issues
- **Flow not fusing**: check `cs_in_air`, `SENS_FLOW_MINHGT`, and range validity.
- **Slow position recovery**: increase `MPC_ACC_HOR_MAX` / `MPC_XY_VEL_P_ACC` carefully.
- **High eph**: verify `EKF2_OF_DELAY`, flow quality, and range noise.

## Startup/Shutdown
### Simulation
1. Start SITL Gazebo simulation.
2. Start MAVROS.
3. After the sim is fully up, run `./scripts/simulation.sh config_sitl.json`.
4. Stop with `Ctrl+C` after the simulation ends.

### Real Hardware
1. `executor.sh` is installed as a systemd service on the Raspberry Pi.
2. The service starts on power‑up and stops on power‑down (or via `systemctl stop`).

## Operational Notes
- No explicit failsafe in this pipeline.
- Optical flow is not fused while the vehicle is detected as on‑ground.
- Above ~70 m AGL, optical‑flow stability degrades and oscillations may appear.
- Goal: Position mode should hold without GPS using optical flow + range.
