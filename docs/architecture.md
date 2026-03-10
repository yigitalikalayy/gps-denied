# Architecture

## Overview
This repository implements a GPS‑denied navigation pipeline based on optical flow, IMU gyro compensation, and range sensing. The system runs on a companion computer (e.g., Raspberry Pi) and publishes MAVLink `OPTICAL_FLOW_RAD` and `DISTANCE_SENSOR` messages to PX4. EKF2 fuses these measurements to estimate horizontal velocity/position and height‑above‑ground‑level (HAGL) without GNSS.

## Data Flow
1. **Camera** captures grayscale frames (ROS or PiCamera backends).
2. **Optical flow** estimates per‑frame pixel motion via KLT/FAST tracking with outlier rejection and (optional) adaptive RANSAC thresholds.
3. **Gyro compensation** integrates body angular rates over the same interval to correct rotational flow components.
4. **Rangefinder** provides ground distance; filtering (median/EMA/jump limits) stabilizes the measurement.
5. **Time base alignment** maps frame time to FCU boot time (and optional motion‑based time sync) to keep EKF timestamps consistent.
6. **MAVLink output** publishes `OPTICAL_FLOW_RAD` and `DISTANCE_SENSOR` at the configured rate.

## Core Components
- `src/algorithms/optical_flow/`
  - `camera.py`: frame acquisition + camera calibration interface.
  - `flow_estimator.py`: KLT/FAST tracking, outlier filtering.
  - `time_sync.py`: motion‑based time alignment (optional).
  - `main.py`: pipeline orchestration and MAVLink publishing.
  - `flight_logger.py`: per‑flight CSV logging.
- `src/sensors/`
  - `imu.py`: IMU backends (if enabled).
  - `lidar.py`: LW20 ASCII/Binary backends.
- `src/mavlink_bridge/`
  - `mavlink_bridge.py`: MAVLink I/O, IMU stream request, gyro/attitude parsing.
  - `mavlink_messages.py` / `mavlink_v1.py`: message packing and parsing.

## Configuration
- `config.json`: real hardware defaults.

Key parameters:
- `flow.*`: feature tracking, RANSAC thresholds, optical flow scaling.
- `gyro.*`: axis swap/signs and gyro‑flow alignment.
- `lidar.*`: backend + filtering parameters.
- `time_sync.*`: motion‑based timestamp alignment (optional).
- `logging.flight_log.*`: per‑run CSV log output.

## Logging
Each run creates a CSV log containing:
- Camera timing and feature stats.
- Flow vectors and quality.
- Gyro rates and integrated gyro angles.
- Range data and delays.

Analysis scripts in `analysis/` compute drift/noise/feature decay metrics from these logs.
