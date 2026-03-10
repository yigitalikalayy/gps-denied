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
