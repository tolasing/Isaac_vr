"""
GR00T pick-and-place eval — Franka Panda in Isaac Sim.

Uses ManagerBasedRLEnv with built-in IK-Rel action space (no manual Jacobian).
Scene: white table with legs, robot on top, 30x20x20cm cardboard box, 8cm cube, overhead camera.

Start GR00T policy server first (in gr00t container):
  docker run --rm --gpus all --ipc=host --network host \\
    -e HF_TOKEN=<token> \\
    -v /root/Isaac-GR00T/checkpoints:/checkpoints \\
    -v /root/Isaac-GR00T/modality_human_hand.py:/workspace/modality_human_hand.py \\
    gr00t:latest \\
    python /workspace/gr00t/eval/run_gr00t_server.py \\
      --model-path /checkpoints/groot_pick_place/checkpoint-2000 \\
      --embodiment-tag new_embodiment \\
      --modality-config-path /workspace/modality_human_hand.py \\
      --port 5555

Run inside isaac-lab-base-gui container:
  /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/eval_groot_franka.py \\
    --headless --enable_cameras --num_episodes 50
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T Franka pick-and-place eval")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--max_steps",    type=int, default=300)
parser.add_argument("--server_host",  type=str, default="localhost")
parser.add_argument("--server_port",  type=int, default=5555)
parser.add_argument("--n_action_steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── after Isaac Sim is up ─────────────────────────────────────────────────────

import io
import time
import numpy as np
import torch
import msgpack
import zmq

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
import isaaclab_tasks.manager_based.manipulation.lift  # noqa: registers envs

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg


# ── GR00T policy client ───────────────────────────────────────────────────────

def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(type(obj))


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class PolicyClient:
    def __init__(self, host="localhost", port=5555):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt(zmq.RCVTIMEO, 15_000)

    def get_action(self, obs: dict) -> list:
        data = {"observation": obs, "options": {}}
        self._sock.send(msgpack.packb({"endpoint": "get_action", "data": data},
                                      default=_encode, use_bin_type=True))
        return msgpack.unpackb(self._sock.recv(), object_hook=_decode, raw=False)

    def ping(self) -> bool:
        try:
            self._sock.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
            self._sock.recv()
            return True
        except zmq.error.Again:
            return False

    def close(self):
        self._sock.close()
        self._ctx.term()


# ── Scene dimensions ──────────────────────────────────────────────────────────

TABLE_TOP_W = 1.20
TABLE_TOP_D = 0.80
TABLE_TOP_T = 0.04
LEG_H       = 0.71
LEG_W       = 0.05
SURFACE_Z   = LEG_H + TABLE_TOP_T          # 0.75 m
TABLE_CX    = 0.50

BOX_L, BOX_W, BOX_H = 0.30, 0.20, 0.20
WALL_T    = 0.008
CUBE_SIZE = 0.08

BOX_X,  BOX_Y  = 0.60,  0.10
CUBE_X, CUBE_Y = 0.45, -0.15
BOX_POS  = np.array([BOX_X,  BOX_Y,  SURFACE_Z])   # top of table
CUBE_POS = np.array([CUBE_X, CUBE_Y, SURFACE_Z + CUBE_SIZE / 2])


def _white():
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95))

def _brown():
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.53, 0.27))


# ── Custom scene ──────────────────────────────────────────────────────────────

@configclass
class PickPlaceSceneCfg(ObjectTableSceneCfg):
    """Override the default scene: custom white table + cardboard box + cube + camera."""

    # Disable parent's USD-based table (replaced by primitive table_top + legs below)
    table = None

    # ── EEF frame transformer (required by LiftEnvCfg) ────────────────────────
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # ── White table with 4 legs ───────────────────────────────────────────────
    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX, 0.0, LEG_H + TABLE_TOP_T / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_TOP_W, TABLE_TOP_D, TABLE_TOP_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )
    leg_fl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegFL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX - TABLE_TOP_W/2 + LEG_W/2, -TABLE_TOP_D/2 + LEG_W/2, LEG_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_white()),
    )
    leg_fr = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegFR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX - TABLE_TOP_W/2 + LEG_W/2, TABLE_TOP_D/2 - LEG_W/2, LEG_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_white()),
    )
    leg_bl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegBL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX + TABLE_TOP_W/2 - LEG_W/2, -TABLE_TOP_D/2 + LEG_W/2, LEG_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_white()),
    )
    leg_br = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegBR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX + TABLE_TOP_W/2 - LEG_W/2, TABLE_TOP_D/2 - LEG_W/2, LEG_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_white()),
    )

    # ── Cardboard box — 5 static panels ───────────────────────────────────────
    box_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxBottom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y, SURFACE_Z + WALL_T / 2]
        ),
        spawn=sim_utils.CuboidCfg(size=(BOX_L, BOX_W, WALL_T),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_brown()),
    )
    box_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxFrontWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y - BOX_W/2 + WALL_T/2, SURFACE_Z + BOX_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(BOX_L, WALL_T, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_brown()),
    )
    box_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxBackWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y + BOX_W/2 - WALL_T/2, SURFACE_Z + BOX_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(BOX_L, WALL_T, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_brown()),
    )
    box_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxLeftWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X - BOX_L/2 + WALL_T/2, BOX_Y, SURFACE_Z + BOX_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(WALL_T, BOX_W, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_brown()),
    )
    box_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxRightWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X + BOX_L/2 - WALL_T/2, BOX_Y, SURFACE_Z + BOX_H/2]
        ),
        spawn=sim_utils.CuboidCfg(size=(WALL_T, BOX_W, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_brown()),
    )

    # ── Camera — 45° angled overhead, 1280×720 ────────────────────────────────
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.04,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, -0.9, SURFACE_Z + 1.0),
            rot=(0.7071, -0.4082, 0.4082, 0.4082),
            convention="world",
        ),
    )


# ── Custom env config ─────────────────────────────────────────────────────────

@configclass
class PickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Replace scene with our custom one
        self.scene = PickPlaceSceneCfg(num_envs=1, env_spacing=5.0)

        # Override default cube with our primitive 8cm cube on the table surface
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS.tolist()),
        )

        # Place robot on table surface
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, SURFACE_Z),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6":  3.037,
                    "panda_joint7":  0.785,
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )

        # IK-Rel action: 6D pose delta (we'll zero the rotation component)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Reduce episode length for eval
        self.episode_length_s = 6.0


# ── Success check ─────────────────────────────────────────────────────────────

def cube_inside_box(cube_pos: np.ndarray) -> bool:
    x_ok = abs(cube_pos[0] - BOX_X) < (BOX_L / 2 - CUBE_SIZE / 2)
    y_ok = abs(cube_pos[1] - BOX_Y) < (BOX_W / 2 - CUBE_SIZE / 2)
    z_ok = (SURFACE_Z + WALL_T) < cube_pos[2] < (SURFACE_Z + BOX_H - CUBE_SIZE / 2)
    return bool(x_ok and y_ok and z_ok)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env_cfg = PickPlaceEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    from isaaclab.sensors import Camera
    camera: Camera = env.scene["camera"]

    # Connect to policy server
    client = PolicyClient(host=args_cli.server_host, port=args_cli.server_port)
    print("Waiting for GR00T policy server...", flush=True)
    while not client.ping():
        time.sleep(1.0)
    print("Connected to GR00T policy server.", flush=True)

    # Action space:
    #   arm:     6D pose delta [dx,dy,dz, dRx,dRy,dRz]  (scale=0.5)
    #   gripper: 1D binary [-1=close, +1=open]
    # Total: 7D
    device = env.device
    successes = 0

    for ep in range(args_cli.num_episodes):
        obs, _ = env.reset()
        action_queue: list[np.ndarray] = []

        for step in range(args_cli.max_steps):
            # ── Observe ──────────────────────────────────────────────────────
            camera.update(env.sim.get_physics_dt())
            rgb = camera.data.output["rgb"][0].cpu().numpy()  # (H,W,3) uint8

            # EEF state from frame transformer
            eef_pos = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()  # (3,)
            gripper_pos = env.scene["robot"].data.joint_pos[0, -2].item()           # finger_joint1

            gripper_state = np.array([[gripper_pos]], dtype=np.float32)

            # ── Query GR00T ───────────────────────────────────────────────────
            if len(action_queue) == 0:
                obs_dict = {
                    "video": {"top": rgb[np.newaxis, np.newaxis]},            # (1,T=1,H,W,3)
                    "state": {
                        "single_arm": eef_pos[np.newaxis, np.newaxis],        # (1,T=1,3)
                        "gripper":    gripper_state[:, np.newaxis],            # (1,T=1,1)
                    },
                    "language": {
                        "annotation.human.task_description": [
                            ["pick up the cube and place it inside the cardboard box"]
                        ]
                    },
                }
                result = client.get_action(obs_dict)
                action_dict = result[0] if isinstance(result, list) else result
                arm_act  = np.array(action_dict["single_arm"], dtype=np.float32)[0]  # (H,3)
                grip_act = np.array(action_dict["gripper"],    dtype=np.float32)[0]  # (H,1)

                # ── COORD-BUG DIAGNOSTIC ──────────────────────────────────────
                # Check if model action points toward or away from cube.
                # cos_sim ≈ -1 means opposite direction → calibration M.T bug confirmed.
                to_cube = CUBE_POS - eef_pos
                a0 = arm_act[0]
                if np.linalg.norm(a0) > 1e-6 and np.linalg.norm(to_cube) > 1e-6:
                    cos_sim = float(np.dot(to_cube, a0) /
                                    (np.linalg.norm(to_cube) * np.linalg.norm(a0)))
                    print(f"  [diag] eef={np.round(eef_pos,3)}  "
                          f"action={np.round(a0,4)}  cos_sim_to_cube={cos_sim:.3f}", flush=True)

                # ── WORKAROUND: undo M.T vs M calibration bug ────────────────
                # M_buggy = M_true.T, so d_true = R^2 @ d_buggy = [-dy, -dz, dx]
                # x was already correct sign; y and z needed negation + axis swap.
                arm_act = np.stack(
                    [-arm_act[:, 1], -arm_act[:, 2], arm_act[:, 0]], axis=-1
                )

                # Build 7D action: [dx,dy,dz,0,0,0, gripper_binary]
                rot_zeros = np.zeros_like(arm_act)                                    # (H,3)
                grip_binary = np.where(grip_act > 0, 1.0, -1.0)                      # (H,1)
                full_act = np.concatenate([arm_act, rot_zeros, grip_binary], axis=-1) # (H,7)
                action_queue = list(full_act[: args_cli.n_action_steps])

            delta = action_queue.pop(0)  # (7,)
            action_t = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)

            obs, _, terminated, truncated, _ = env.step(action_t)

            # ── Check success ─────────────────────────────────────────────────
            cube_pos = env.scene["object"].data.root_pos_w[0].cpu().numpy()
            if cube_inside_box(cube_pos) or terminated.any() or truncated.any():
                success = cube_inside_box(cube_pos)
                if success:
                    successes += 1
                print(f"Episode {ep+1:3d}/{args_cli.num_episodes}  success={success}  "
                      f"sr={successes/(ep+1):.2f}", flush=True)
                break
        else:
            success = cube_inside_box(
                env.scene["object"].data.root_pos_w[0].cpu().numpy()
            )
            if success:
                successes += 1
            print(f"Episode {ep+1:3d}/{args_cli.num_episodes}  success={success}  "
                  f"sr={successes/(ep+1):.2f}", flush=True)

    print(f"\n=== Final success rate: {successes}/{args_cli.num_episodes} "
          f"= {successes/args_cli.num_episodes:.1%} ===")
    client.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
