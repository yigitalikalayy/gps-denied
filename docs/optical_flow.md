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
