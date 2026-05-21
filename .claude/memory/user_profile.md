---
name: user-profile
description: "Who the user is — Ganesh Tolasing, robotics/XR engineer working on Isaac Lab VR teleoperation with Meta Quest + NVIDIA CloudXR"
metadata: 
  node_type: memory
  type: user
  originSessionId: 0e569bc8-a2d7-4a62-98c5-40ce17bb07d3
---

**Name**: Ganesh Tolasing (git user: Ganesh Tolasing, email: tolasingganesh@gmail.com)

**Role**: Robotics / XR engineer working on Isaac Lab-based VR teleoperation systems.

**Stack**: Isaac Lab (IsaacSim 5.1 / Isaac Lab 3.0), NVIDIA CloudXR 6.1.0, Meta Quest VR headset, IsaacTeleop package (`isaacteleop[retargeters,cloudxr]~=1.0.0`), Docker container (`isaac-lab-base`).

**Tasks being built**:
- `Isaac-Stack-Cube-Franka-IK-Abs-v0` — Franka arm stacking task, right controller drives EE via Se3AbsRetargeter, launched via `teleop_se3_agent.py`
- `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` — G1 humanoid locomanipulation, VR motion controllers drive wrist pose + TriHand gripper, launched via `record_demos.py`

**Working directory**: `/root/groot` (host), `/workspace/isaaclab` (inside container `isaac-lab-base`)

**Isaac-sim Python**: `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3` (Python 3.12)

**Preferences**:
- Concise responses, no trailing summaries
- Does not want unnecessary packages installed on host
- Explicit "push to git only when I say so"
- Prefers to understand what's being changed before approving
