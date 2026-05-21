---
name: project-xr-teleop
description: "Summary of all XR teleoperation work done — CloudXR setup, rendering fixes, controller tracking, anchor calibration, se3 retargeter patch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e569bc8-a2d7-4a62-98c5-40ce17bb07d3
---

## Infrastructure

**CloudXR startup**
- Start: `unset DISPLAY && ./isaaclab.sh -p -m isaacteleop.cloudxr --accept-eula` (run inside container)
- Ready signal: wait for `/root/.cloudxr/run/runtime_started` (NOT just `cloudxr.env`)
- Kill stale runs: `ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9` + `rm -f /root/.cloudxr/run/cloudxr.pid /root/.cloudxr/run/runtime_started`

**Sim launch (Franka stack)**:
```
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Stack-Cube-Franka-IK-Abs-v0 --device cuda
```

**Sim launch (G1 locomanipulation)**:
```
./isaaclab.sh -p scripts/environments/teleoperation/record_demos.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --device cuda
```

## Bugs Fixed

**daqp version mismatch** (arm not moving at all):
- `qpsolvers` 4.12.0 passes `primal_start=` kwarg to `daqp.solve()`, but `daqp` 0.7.2 doesn't support it
- Fix: `pip install 'daqp==0.8.5'` inside container
- File: `source/isaaclab/isaaclab/controllers/pink_ik/pink_ik.py` (uses daqp solver)

**Se3AbsRetargeter position offset applied in wrist frame** (arm jumps when holding still):
- Bug: `position = position + base_rot.apply(self._target_offset_pos)` — large offsets amplify rotational jitter
- Fix: patched to `position = position + self._target_offset_pos` (world frame)
- File patched: `/isaac-sim/kit/python/lib/python3.12/site-packages/isaacteleop/retargeters/se3_retargeter.py:276` inside container

## Rendering Changes (committed + pushed to `xr` branch)

**`apps/isaaclab.python.xr.openxr.kit`**:
- `rtx.rendermode = "RasterizationLighting"` (was `"RaytracedLighting"`) — eliminates 10-min shader compile, GPU 74%→47%, temp 63°C→57°C
- `rtx.shadows.enabled = false`
- Hand tracking extensions disabled (`omni.kit.xr.openxr.ext.hand_tracking`, `isaacsim.xr.openxr.hand_tracking` → `false`)

**`apps/isaaclab.python.xr.openxr.headless.kit`**:
- Added performance rendering settings block

## G1 Locomanipulation Pipeline (`locomanipulation_g1_env_cfg.py`)

- Switched from `HandsSource` + custom `HandTrackingFingerRetargeter` → `ControllersSource` + `TriHandMotionControllerRetargeter`
- 32D action tensor: [left_wrist(7), right_wrist(7), hand_joints(14), locomotion(4)]
- XR config: `anchor_pos=(0.0, 0.0, -0.95)`, `anchor_prim_path="/World/envs/env_0/Robot/pelvis"`, `FOLLOW_PRIM_SMOOTHED`

## Franka Stack XR Config (currently reverted to git HEAD — no overrides)

The file `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_abs_env_cfg.py` was iterated on heavily during calibration. Currently at git HEAD (no XR overrides). Key learnings:

**anchor_pos semantics**: The sim point at `anchor_pos` coordinates appears at the user's VR floor origin. E.g. `(-1.5, 0, -1.05)` = user stands 1.5m behind robot, VR floor = sim floor.

**anchor_rot = (0, 0, -0.707, 0.707)** = -90° Z rotation. Maps:
- VR forward → sim +X (toward table) ✓
- VR right → sim -Y (robot's right) ✓
- VR up → sim +Z ✓

**Workspace**: table at sim X=0.5, robot at origin, EE working height ≈ Z=0.35, ground at Z=-1.05.

**target_offset_x/y/z** in Se3RetargeterConfig: after the wrist-frame patch, these are world-frame constants. Large values (>0.5m) are needed when user is far from workspace. Were causing arm to float left when offset_y was non-zero.

**render_interval = 1** needed (not 2) to avoid camera shake in VR.
**near_plane = 0.05** needed for close-up robot viewing.

## XrCfg Key Defaults
- `fixed_anchor_height = True`
- `anchor_rotation_mode = XrAnchorRotationMode.FIXED`
- `near_plane = 0.15` (default — too far for close robot work, override to 0.05)
