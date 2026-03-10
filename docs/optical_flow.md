# Optical Flow Pipeline

## Algorithm Summary
1. **Preprocessing**: optional CLAHE + gamma correction on grayscale frames.
2. **Feature detection**: Shi‑Tomasi or FAST, with border masking.
3. **Tracking**: pyramidal Lucas‑Kanade (KLT) between consecutive frames.
4. **Outlier rejection**:
   - Forward‑backward check (optional)
   - MAD‑based residual filtering
   - RANSAC affine model
5. **Flow estimation**: robust mean of inlier displacements.
6. **Quality metric**: based on inlier ratio, MAD, and motion magnitude.

## Outputs
- `flow_px_x/y`: pixel displacement per frame
- `flow_rad_x/y`: pixel displacement divided by focal length
- `integrated_*`: time‑integrated flow/gyro values for `OPTICAL_FLOW_RAD`

## Tuning Tips
- **Low texture**: increase feature count or lower `feature_quality`.
- **Blur / fast motion**: raise RANSAC threshold slightly.
- **Too many outliers**: lower RANSAC threshold or increase MAD rejection.

## Frame Alignment & Calibration
The pipeline relies on consistent axis mapping between camera flow, IMU gyro, and PX4 body frame.

**Camera frame (image) conventions**
- Image X: right, Image Y: down (standard image coordinates).
- Downward‑facing camera: forward motion produces downward pixel motion; rightward motion produces leftward pixel motion.

**Alignment controls (config)**
- `flow.axis_swap_xy`: swap flow axes if camera is rotated 90° relative to body.
- `flow.axis_sign_x`, `flow.axis_sign_y`: flip flow signs to match PX4 body frame expectations.
- `gyro.axis_swap_xy`, `gyro.axis_sign_*`: map IMU gyro to the same body frame.
- `gyro.frame`: set to `ros_flu` (ROS IMU) or `px4_frd` as appropriate.

**Calibration**
- Intrinsics are supplied via `camera.calibration` in `config_sitl.json` (precomputed) or via cache file in `config.json` (real camera).
- Any change in lens or mounting angle requires updating focal length and alignment signs.

**Quick validation**
1. Move the vehicle forward: flow should be consistent with forward motion.
2. Move right: flow should be consistent with rightward motion.
3. Yaw in place: flow should be dominated by gyro compensation.
If signs are inverted, adjust `axis_sign_*` and re‑test.
