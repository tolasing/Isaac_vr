#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Generate honeycomb_paper.usd — 10×10×3 cm honeycomb paper pad.

Physics: single rigid body with a simple box collider.
Visual:  kraft-brown box body + raised hexagonal wall ridges on the top face.

Hex cell layout (flat-top orientation):
  circumradius  s = 8 mm  → cell diameter ~16 mm
  wall height     3 mm above the top face
  wall thickness  1 mm
"""

import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_material(stage, parent_path, name, color):
    mat_path = parent_path.AppendChild(name)
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _hex_unique_edges(W, D, s):
    """Return all unique edges of a flat-top hexagonal grid over the W×D area.

    Centers are generated with a one-cell border so edges near the boundary are
    included.  Deduplication uses rounded integer keys (0.1 mm precision).
    """
    dx = 1.5 * s
    dy = math.sqrt(3) * s

    centers = []
    col = 0
    x = -W / 2 - s
    while x <= W / 2 + s:
        y_off = dy / 2 if col % 2 == 1 else 0.0
        y = -D / 2 - s + y_off
        while y <= D / 2 + s:
            centers.append((x, y))
            y += dy
        x += dx
        col += 1

    def corners(cx, cy):
        return [
            (cx + s * math.cos(i * math.pi / 3), cy + s * math.sin(i * math.pi / 3))
            for i in range(6)
        ]

    seen = {}
    for cx, cy in centers:
        verts = corners(cx, cy)
        for i in range(6):
            v1, v2 = verts[i], verts[(i + 1) % 6]
            k1 = (round(v1[0] * 10000), round(v1[1] * 10000))
            k2 = (round(v2[0] * 10000), round(v2[1] * 10000))
            key = (min(k1, k2), max(k1, k2))
            if key not in seen:
                seen[key] = (v1, v2)
    return list(seen.values())


def _build_honeycomb_mesh(stage, mesh_path, W, D, z_base, wall_h, wall_t, hex_size, material):
    """Build a UsdGeom.Mesh of raised hexagonal wall ridges on a flat face.

    Each unique hex edge becomes a thin box prism:
      - two long faces (front / back)
      - top face
      - two end caps
    The mesh is double-sided so winding order doesn't matter for normals.
    """
    edges = _hex_unique_edges(W, D, hex_size)
    hw = wall_t / 2
    z0, z1 = z_base, z_base + wall_h
    # Only keep edges where both endpoints sit inside (or just outside) the pad
    margin = hex_size * 0.6

    pts, counts, idxs = [], [], []

    def quad(a, b, c, d):
        base = len(pts)
        pts.extend([Gf.Vec3f(*v) for v in (a, b, c, d)])
        counts.append(4)
        idxs.extend([base, base + 1, base + 2, base + 3])

    for v1, v2 in edges:
        if (
            min(v1[0], v2[0]) < -W / 2 - margin
            or max(v1[0], v2[0]) > W / 2 + margin
            or min(v1[1], v2[1]) < -D / 2 - margin
            or max(v1[1], v2[1]) > D / 2 + margin
        ):
            continue

        edx, edy = v2[0] - v1[0], v2[1] - v1[1]
        ln = math.sqrt(edx * edx + edy * edy)
        if ln < 1e-9:
            continue
        # unit perpendicular (left of edge direction)
        px, py = -edy / ln, edx / ln

        b0 = (v1[0] + hw * px, v1[1] + hw * py, z0)
        b1 = (v2[0] + hw * px, v2[1] + hw * py, z0)
        b2 = (v2[0] - hw * px, v2[1] - hw * py, z0)
        b3 = (v1[0] - hw * px, v1[1] - hw * py, z0)
        t0 = (v1[0] + hw * px, v1[1] + hw * py, z1)
        t1 = (v2[0] + hw * px, v2[1] + hw * py, z1)
        t2 = (v2[0] - hw * px, v2[1] - hw * py, z1)
        t3 = (v1[0] - hw * px, v1[1] - hw * py, z1)

        quad(b0, b1, t1, t0)  # front long face
        quad(b3, t3, t2, b2)  # back long face
        quad(t0, t1, t2, t3)  # top face
        quad(b3, b0, t0, t3)  # left end cap
        quad(b1, b2, t2, t1)  # right end cap

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idxs)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# ── main ──────────────────────────────────────────────────────────────────────


def create_honeycomb_paper(output_path):
    W, D, H = 0.10, 0.10, 0.03   # 10×10×3 cm
    hex_size = 0.008              # circumradius → ~16 mm cell diameter
    wall_h = 0.003                # 3 mm ridges above top face
    wall_t = 0.001                # 1 mm wall thickness

    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_path = Sdf.Path("/HoneycombPaper")
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.04)  # 40 g

    body_mat = _make_material(stage, root_path, "KraftMat", (0.72, 0.55, 0.30))
    ridge_mat = _make_material(stage, root_path, "RidgeMat", (0.82, 0.65, 0.40))

    # ── Box body: collision + visual ─────────────────────────────────────────
    body_path = root_path.AppendChild("Body")
    body = UsdGeom.Cube.Define(stage, body_path)
    body.CreateSizeAttr(1.0)
    UsdGeom.Xformable(body.GetPrim()).AddScaleOp().Set(Gf.Vec3f(W, D, H))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(body_mat)

    # ── Honeycomb ridge mesh on top face ─────────────────────────────────────
    _build_honeycomb_mesh(
        stage,
        root_path.AppendChild("TopPattern"),
        W, D,
        z_base=H / 2,
        wall_h=wall_h,
        wall_t=wall_t,
        hex_size=hex_size,
        material=ridge_mat,
    )

    stage.Save()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "honeycomb_paper.usd")
    create_honeycomb_paper(out)
