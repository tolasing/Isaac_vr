#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Generate cardboard_box.usd — 30x20x15 cm articulated box with hinged top flap.

Box dimensions (local frame, centered at origin):
  x: -0.15 to +0.15  (30 cm, length)
  y: -0.10 to +0.10  (20 cm, width — widest face)
  z: -0.075 to +0.075 (15 cm, height)

Flap: covers the full top face (30x20 cm), hinged along the back edge (y=-0.10).
Hinge axis: X.  0° = closed (flat), 120° = wide open.
No spring drive — flap starts fully open and is free to move under gravity and robot contact.
Gravity holds the flap at the 120° upper limit (COM past 90° = stable against limit).
"""

import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _add_box_wall(stage, parent_path, name, center, half_extents, material):
    """Create a scaled UsdGeom.Cube wall with collision and material binding."""
    prim_path = parent_path.AppendChild(name)
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)

    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(half_extents[0] * 2.0, half_extents[1] * 2.0, half_extents[2] * 2.0))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def _make_cardboard_material(stage, parent_path):
    """Create a simple brown UsdPreviewSurface material for cardboard look."""
    mat_path = parent_path.AppendChild("CardboardMaterial")
    material = UsdShade.Material.Define(stage, mat_path)

    shader_path = mat_path.AppendChild("PreviewSurface")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.65, 0.45, 0.22))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_cardboard_box(output_path):
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    t = 0.003  # wall thickness 3 mm

    # ── Root / ArticulationRoot ───────────────────────────────────────────────
    root_path = Sdf.Path("/CardboardBox")
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    stage.SetDefaultPrim(root.GetPrim())

    material = _make_cardboard_material(stage, root_path)

    # ── BoxBase: 5-sided shell (bottom + 4 walls, open top) ──────────────────
    base_path = root_path.AppendChild("BoxBase")
    UsdGeom.Xform.Define(stage, base_path)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(base_path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(base_path)).CreateMassAttr(10.5)

    # Box spans x∈[-0.15,0.15], y∈[-0.10,0.10], z∈[-0.075,0.075]
    _add_box_wall(stage, base_path, "Bottom",    (0, 0, -0.075 + t / 2), (0.15, 0.10, t / 2),     material)
    _add_box_wall(stage, base_path, "FrontWall", (0,  0.10 - t / 2, 0),  (0.15, t / 2, 0.075),    material)
    _add_box_wall(stage, base_path, "BackWall",  (0, -0.10 + t / 2, 0),  (0.15, t / 2, 0.075),    material)
    _add_box_wall(stage, base_path, "LeftWall",  (-0.15 + t / 2, 0, 0),  (t / 2, 0.10 - t, 0.075), material)
    _add_box_wall(stage, base_path, "RightWall", ( 0.15 - t / 2, 0, 0),  (t / 2, 0.10 - t, 0.075), material)

    # ── Flap: top panel, pre-positioned fully open at 120° ───────────────────
    flap_path = root_path.AppendChild("Flap")
    flap_xf = UsdGeom.Xform.Define(stage, flap_path)

    # Hinge is at (0, -0.10+t, 0.075). Offset from hinge to flap centre at 0° (closed):
    #   dy = 0.10-t, dz = t/2.  After rotating 120° about X:
    hinge_y = -0.10 + t
    dy0, dz0 = 0.10 - t, t / 2
    theta = math.radians(120.0)
    flap_y = hinge_y + dy0 * math.cos(theta) - dz0 * math.sin(theta)
    flap_z = 0.075 + dy0 * math.sin(theta) + dz0 * math.cos(theta)
    UsdGeom.XformCommonAPI(flap_xf).SetTranslate(Gf.Vec3d(0.0, flap_y, flap_z))
    UsdGeom.XformCommonAPI(flap_xf).SetRotate(Gf.Vec3f(120.0, 0.0, 0.0))

    UsdPhysics.RigidBodyAPI.Apply(flap_xf.GetPrim())
    UsdPhysics.MassAPI.Apply(flap_xf.GetPrim()).CreateMassAttr(0.1)

    _add_box_wall(stage, flap_path, "Panel", (0.0, 0.0, 0.0), (0.15, 0.10, t / 2), material)

    # ── Revolute joint: hinge at back edge of top opening ────────────────────
    joint_path = root_path.AppendChild("FlapJoint")
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([flap_path])
    joint.CreateAxisAttr("X")

    # Pivot in BoxBase frame: back top edge = (0, -0.10+t, 0.075)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, -0.10 + t, 0.075))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Pivot in Flap frame: back edge of panel = (0, -0.10, -t/2) in panel-centered coords
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, -0.10, -t / 2))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Angular limits in degrees: 0=closed, 120=wide open
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(120.0)

    # Damping only — no spring. Prevents bouncing at limits without fighting robot/gravity.
    drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateTargetPositionAttr(0.0)
    drive_api.CreateStiffnessAttr(0.0)
    drive_api.CreateDampingAttr(0.5)
    drive_api.CreateMaxForceAttr(2.0)

    stage.Save()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "cardboard_box.usd")
    create_cardboard_box(out)
