"""
Quick scene test — no policy server needed.

Scene: white table with legs, Franka Panda mounted on top,
cardboard box (30x20x20cm) and cube (8cm) on the table surface.

Run inside isaac-lab-base-gui container:
  /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/test_scene.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scene smoke test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── after Isaac Sim is up ─────────────────────────────────────────────────────

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


# ── Dimensions ────────────────────────────────────────────────────────────────
TABLE_TOP_W  = 1.20   # table width  (X)
TABLE_TOP_D  = 0.80   # table depth  (Y)
TABLE_TOP_T  = 0.04   # tabletop thickness
LEG_H        = 0.71   # leg height
LEG_W        = 0.05   # leg cross-section
SURFACE_Z    = LEG_H + TABLE_TOP_T          # 0.75 m — top of table surface

BOX_L, BOX_W, BOX_H = 0.30, 0.20, 0.20
WALL_T    = 0.008
CUBE_SIZE = 0.08

# Positions on the table surface (objects + robot all at SURFACE_Z)
ROBOT_X, ROBOT_Y = 0.0,  0.0     # robot base at near edge of table
BOX_X,   BOX_Y   = 0.60, 0.10   # cardboard box
CUBE_X,  CUBE_Y  = 0.45, -0.15  # small cube

# Table centre (for placing top + legs)
TABLE_CX, TABLE_CY = 0.50, 0.00


def _white():
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95))

def _brown():
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.53, 0.27))

def _static_box(size, pos, material):
    return AssetBaseCfg(
        prim_path="PLACEHOLDER",   # overridden per-field
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(pos)),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=material,
        ),
    )


@configclass
class TestSceneCfg(InteractiveSceneCfg):

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )

    # ── Table top ─────────────────────────────────────────────────────────────
    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX, TABLE_CY, LEG_H + TABLE_TOP_T / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_TOP_W, TABLE_TOP_D, TABLE_TOP_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )

    # ── Four legs (flat prim names) ────────────────────────────────────────────
    leg_fl = AssetBaseCfg(   # front-left
        prim_path="{ENV_REGEX_NS}/LegFL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX - TABLE_TOP_W / 2 + LEG_W / 2,
                 TABLE_CY - TABLE_TOP_D / 2 + LEG_W / 2,
                 LEG_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )
    leg_fr = AssetBaseCfg(   # front-right
        prim_path="{ENV_REGEX_NS}/LegFR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX - TABLE_TOP_W / 2 + LEG_W / 2,
                 TABLE_CY + TABLE_TOP_D / 2 - LEG_W / 2,
                 LEG_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )
    leg_bl = AssetBaseCfg(   # back-left
        prim_path="{ENV_REGEX_NS}/LegBL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX + TABLE_TOP_W / 2 - LEG_W / 2,
                 TABLE_CY - TABLE_TOP_D / 2 + LEG_W / 2,
                 LEG_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )
    leg_br = AssetBaseCfg(   # back-right
        prim_path="{ENV_REGEX_NS}/LegBR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CX + TABLE_TOP_W / 2 - LEG_W / 2,
                 TABLE_CY + TABLE_TOP_D / 2 - LEG_W / 2,
                 LEG_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(LEG_W, LEG_W, LEG_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_white(),
        ),
    )

    # ── Franka Panda — base on table surface ───────────────────────────────────
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ROBOT_X, ROBOT_Y, SURFACE_Z),
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

    # ── Small cube ─────────────────────────────────────────────────────────────
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[CUBE_X, CUBE_Y, SURFACE_Z + CUBE_SIZE / 2]
        ),
    )

    # ── Cardboard box — 5 static panels ───────────────────────────────────────
    box_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxBottom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y, SURFACE_Z + WALL_T / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOX_L, BOX_W, WALL_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_brown(),
        ),
    )
    box_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxFrontWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y - BOX_W / 2 + WALL_T / 2, SURFACE_Z + BOX_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOX_L, WALL_T, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_brown(),
        ),
    )
    box_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxBackWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X, BOX_Y + BOX_W / 2 - WALL_T / 2, SURFACE_Z + BOX_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOX_L, WALL_T, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_brown(),
        ),
    )
    box_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxLeftWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X - BOX_L / 2 + WALL_T / 2, BOX_Y, SURFACE_Z + BOX_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(WALL_T, BOX_W, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_brown(),
        ),
    )
    box_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxRightWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[BOX_X + BOX_L / 2 - WALL_T / 2, BOX_Y, SURFACE_Z + BOX_H / 2]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(WALL_T, BOX_W, BOX_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_brown(),
        ),
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


def main():
    sim_cfg = SimulationCfg(dt=0.01, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, -1.5, 2.5], target=[0.5, 0.0, SURFACE_Z])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    spawn_ground_plane("/World/GroundPlane", GroundPlaneCfg())

    sim.reset()
    print("[test_scene] Scene loaded OK")

    robot: Articulation = scene["robot"]
    cube: RigidObject   = scene["cube"]
    camera: Camera      = scene["camera"]

    home = torch.tensor(
        [[0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785, 0.04, 0.04]],
        device=sim.device,
    )

    for step in range(200):
        robot.set_joint_position_target(home)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

        if step == 100:
            camera.update(sim_cfg.dt)
            rgb = camera.data.output["rgb"]
            cube_pos = cube.data.root_pos_w[0].cpu().numpy()
            eef_ids, _ = robot.find_bodies("panda_hand")
            eef_pos = robot.data.body_pos_w[0, eef_ids[0]].cpu().numpy()

            print(f"[test_scene] Step {step}")
            print(f"  Camera RGB shape : {rgb.shape}  dtype={rgb.dtype}")
            print(f"  Cube position    : {np.round(cube_pos, 3)}")
            print(f"  EEF position     : {np.round(eef_pos, 3)}")
            print(f"  Table surface Z  : {SURFACE_Z:.3f}  (expected cube Z ≈ {SURFACE_Z + CUBE_SIZE/2:.3f})")

    print("[test_scene] 200 steps completed — scene is working.")


if __name__ == "__main__":
    main()
    simulation_app.close()
