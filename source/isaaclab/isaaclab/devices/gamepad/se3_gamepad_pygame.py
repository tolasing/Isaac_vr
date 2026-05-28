# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pygame-based gamepad controller for SE(3) control.

Drop-in replacement for Se3Gamepad that reads /dev/input directly via pygame
instead of Carb's input system. Works in headless and livestream modes where
the Omniverse window never has focus.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase, DeviceCfg


# Xbox controller axis / button indices on Linux
_AXIS_LEFT_X = 0    # left stick left(-1) / right(+1)
_AXIS_LEFT_Y = 1    # left stick up(-1) / down(+1)  — inverted
_AXIS_RIGHT_X = 3   # right stick left(-1) / right(+1)
_AXIS_RIGHT_Y = 4   # right stick up(-1) / down(+1) — inverted

_BTN_X = 2          # toggle gripper

# Named buttons accepted by add_callback()
_NAMED_BUTTONS: dict[str, int] = {
    "A": 0, "B": 1, "X": 2, "Y": 3,
    "LB": 4, "RB": 5,
    "BACK": 6, "SELECT": 6,
    "START": 7,
    "L": 9, "R": 10,
}


class Se3GamepadPygame(DeviceBase):
    """Gamepad controller for SE(3) teleoperation using pygame input.

    Reads the gamepad directly from /dev/input via pygame — no Carb window focus
    required. Output format is identical to Se3Gamepad: a 7-D tensor
    [dx, dy, dz, rx, ry, rz, gripper] where gripper is +1 (open) or -1 (close).

    Stick and button bindings (Xbox layout):
        ========================== =============================
        Description                Control
        ========================== =============================
        Move X (fwd / back)        Left stick up / down
        Move Y (left / right)      Left stick left / right
        Move Z (up / down)         Right stick up / down
        Rotate yaw                 Right stick left / right
        Rotate pitch               D-pad up / down
        Rotate roll                D-pad left / right
        Toggle gripper             X button
        ========================== =============================
    """

    def __init__(self, cfg: Se3GamepadPygameCfg):
        super().__init__()

        import pygame
        self._pygame = pygame

        # Use dummy SDL drivers so pygame doesn't fight the Omniverse window/audio
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.dead_zone = cfg.dead_zone
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device

        self._pygame.init()
        self._pygame.joystick.init()
        if self._pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect a gamepad and try again.")
        self._joystick = self._pygame.joystick.Joystick(0)
        self._joystick.init()

        self._close_gripper = False
        self._gripper_btn_prev = False
        # (positive, negative) × (x, y, z, roll, pitch, yaw)
        self._delta_pose_raw = np.zeros((2, 6))
        self._additional_callbacks: dict = {}

    def __del__(self):
        if not hasattr(self, "_pygame"):
            return
        if self._pygame.joystick.get_init():
            self._joystick.quit()
            self._pygame.joystick.quit()

    def __str__(self) -> str:
        name = self._joystick.get_name() if self._joystick else "unknown"
        return f"Se3GamepadPygame [{name}]"

    def reset(self):
        self._close_gripper = False
        self._gripper_btn_prev = False
        self._delta_pose_raw.fill(0.0)

    def add_callback(self, key, func: Callable):
        """Register a callback for a named button or numeric pygame button index.

        Supported string keys: A, B, X, Y, LB, RB, START, BACK, SELECT, L, R.
        Carb GamepadInput enum values are accepted but silently ignored (no-op)
        so the teleop script's try/except path still works cleanly.
        """
        if isinstance(key, str):
            if key in _NAMED_BUTTONS:
                self._additional_callbacks[_NAMED_BUTTONS[key]] = func
            # string keys like "R", "START", "STOP", "RESET" from the teleop script
            # that don't map to a button are intentionally ignored
        elif isinstance(key, int):
            self._additional_callbacks[key] = func
        # Carb enum values (not int/str) are silently dropped — no-op

    def advance(self) -> torch.Tensor:
        # drain pygame event queue so joystick state is current
        self._pygame.event.pump()

        self._delta_pose_raw.fill(0.0)
        self._read_sticks()
        self._read_dpad()
        self._read_buttons()

        delta_pos = self._resolve(self._delta_pose_raw[:, :3])
        delta_rot = self._resolve(self._delta_pose_raw[:, 3:])
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()

        command = np.concatenate([delta_pos, rot_vec])
        if self.gripper_term:
            command = np.append(command, -1.0 if self._close_gripper else 1.0)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _axis(self, idx: int) -> float:
        v = self._joystick.get_axis(idx)
        return v if abs(v) >= self.dead_zone else 0.0

    def _read_sticks(self):
        # Left stick Y: up = -1 → +x, down = +1 → -x
        ly = self._axis(_AXIS_LEFT_Y)
        self._delta_pose_raw[0, 0] = max(-ly, 0.0) * self.pos_sensitivity  # +x
        self._delta_pose_raw[1, 0] = max(ly, 0.0) * self.pos_sensitivity   # -x

        # Left stick X: right = +1 → +y, left = -1 → -y
        lx = self._axis(_AXIS_LEFT_X)
        self._delta_pose_raw[0, 1] = max(lx, 0.0) * self.pos_sensitivity   # +y
        self._delta_pose_raw[1, 1] = max(-lx, 0.0) * self.pos_sensitivity  # -y

        # Right stick Y: up = -1 → +z, down = +1 → -z
        ry = self._axis(_AXIS_RIGHT_Y)
        self._delta_pose_raw[0, 2] = max(-ry, 0.0) * self.pos_sensitivity  # +z
        self._delta_pose_raw[1, 2] = max(ry, 0.0) * self.pos_sensitivity   # -z

        # Right stick X: right = +1 → +yaw, left = -1 → -yaw
        rx = self._axis(_AXIS_RIGHT_X)
        self._delta_pose_raw[0, 5] = max(rx, 0.0) * self.rot_sensitivity   # +yaw
        self._delta_pose_raw[1, 5] = max(-rx, 0.0) * self.rot_sensitivity  # -yaw

    def _read_dpad(self):
        if self._joystick.get_numhats() == 0:
            return
        hx, hy = self._joystick.get_hat(0)

        # D-pad up(hy=+1) → -pitch, down(hy=-1) → +pitch
        scale = self.rot_sensitivity * 0.8
        if hy > 0:
            self._delta_pose_raw[1, 4] = scale  # -pitch
        elif hy < 0:
            self._delta_pose_raw[0, 4] = scale  # +pitch

        # D-pad right(hx=+1) → -roll, left(hx=-1) → +roll
        if hx > 0:
            self._delta_pose_raw[1, 3] = scale  # -roll
        elif hx < 0:
            self._delta_pose_raw[0, 3] = scale  # +roll

    def _read_buttons(self):
        # X button toggles gripper on press (edge-detect)
        btn_x = bool(self._joystick.get_button(_BTN_X))
        if btn_x and not self._gripper_btn_prev:
            self._close_gripper = not self._close_gripper
        self._gripper_btn_prev = btn_x

        # fire any registered callbacks for buttons that are pressed
        for btn_idx, func in self._additional_callbacks.items():
            if self._joystick.get_button(btn_idx):
                func()

    def _resolve(self, raw: np.ndarray) -> np.ndarray:
        """Convert (2, 3) positive/negative raw buffer to signed (3,) command."""
        sign = raw[1] > raw[0]
        out = raw.max(axis=0)
        out[sign] *= -1
        return out


@dataclass
class Se3GamepadPygameCfg(DeviceCfg):
    """Configuration for Se3GamepadPygame. Identical fields to Se3GamepadCfg."""

    gripper_term: bool = True
    dead_zone: float = 0.01
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.6
    class_type: type[DeviceBase] = Se3GamepadPygame
