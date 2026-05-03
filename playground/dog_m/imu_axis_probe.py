from pathlib import Path
import sys
import time
import argparse

import mujoco
import mujoco.viewer
import numpy as np
from etils import epath

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from playground.dog_m import base


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_input_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return path.as_posix()

    project_relative = PROJECT_ROOT / path_str
    if project_relative.exists():
        return project_relative.as_posix()

    playground_relative = PROJECT_ROOT / "playground" / path_str
    if playground_relative.exists():
        return playground_relative.as_posix()

    return path.as_posix()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad * 0.5
    return np.array(
        [np.cos(half), *(axis * np.sin(half))],
        dtype=np.float64,
    )


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


class ImuAxisProbe:
    def __init__(self, model_path: str, angle_deg: float, pulse_rad_s: float):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        mujoco.mj_step(self.model, self.data)

        self.home_qpos = self.model.keyframe("home").qpos.copy()
        self.home_ctrl = self.model.keyframe("home").ctrl.copy()
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

        self.angle_deg = angle_deg
        self.angle_rad = np.deg2rad(angle_deg)
        self.pulse_rad_s = pulse_rad_s
        self.freeze_joints = True
        self.print_every = 15
        self.counter = 0

        self.gyro_id = self.model.sensor("gyro").id
        self.accel_id = self.model.sensor("accelerometer").id
        self.up_id = self.model.sensor("upvector").id
        self.forward_id = self.model.sensor("forwardvector").id
        self.imu_site_id = self.model.site("imu").id

        self.gyro_adr = self.model.sensor_adr[self.gyro_id]
        self.accel_adr = self.model.sensor_adr[self.accel_id]
        self.up_adr = self.model.sensor_adr[self.up_id]
        self.forward_adr = self.model.sensor_adr[self.forward_id]

        self.base_qpos_addr = self.model.jnt_qposadr[
            np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
        ]
        self.base_qvel_addr = self.model.jnt_dofadr[
            np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
        ]

        self._print_help()
        self._print_state("initial")

    def _print_help(self):
        print("")
        print("IMU axis probe controls")
        print("  7 / 8 : roll  - / +   (around +x)")
        print("  9 / 0 : pitch - / +   (around +y)")
        print("  - / = : yaw   - / +   (around +z)")
        print("  z/x   : gyro pulse around x - / +")
        print("  c/v   : gyro pulse around y - / +")
        print("  b/n   : gyro pulse around z - / +")
        print("  r     : reset to home pose")
        print("  f     : toggle freezing leg joints at home ctrl")
        print("  h     : print this help")
        print("")
        print("Expected local body axes")
        print("  +x forward, +y left, +z up")
        print("  positive gyro follows right-hand rule")
        print("")

    def _get_sensor_vec(self, adr: int) -> np.ndarray:
        return np.array(self.data.sensordata[adr : adr + 3], dtype=np.float64)

    def _base_quat(self) -> np.ndarray:
        return np.array(
            self.data.qpos[self.base_qpos_addr + 3 : self.base_qpos_addr + 7],
            dtype=np.float64,
        )

    def _set_base_quat(self, quat: np.ndarray):
        self.data.qpos[self.base_qpos_addr + 3 : self.base_qpos_addr + 7] = normalize_quat(
            quat
        )

    def _reset_pose(self):
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._print_state("reset")

    def _rotate_base_local(self, axis: np.ndarray, sign: float, label: str):
        delta = axis_angle_to_quat(axis, sign * self.angle_rad)
        new_quat = quat_mul(self._base_quat(), delta)
        self._set_base_quat(new_quat)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        direction = "+" if sign > 0 else "-"
        self._print_state(f"{label} {direction}{self.angle_deg:.1f} deg")

    def _pulse_gyro(self, axis_index: int, sign: float, label: str):
        self.data.qvel[:] = 0.0
        self.data.qvel[self.base_qvel_addr + 3 + axis_index] = sign * self.pulse_rad_s
        mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        direction = "+" if sign > 0 else "-"
        self._print_state(f"{label} gyro pulse {direction}{self.pulse_rad_s:.2f} rad/s")

    def _print_state(self, reason: str):
        gyro = self._get_sensor_vec(self.gyro_adr)
        accel = self._get_sensor_vec(self.accel_adr)
        up = self._get_sensor_vec(self.up_adr)
        forward = self._get_sensor_vec(self.forward_adr)
        quat = self._base_quat()
        print("")
        print(f"[{reason}]")
        print(f"base quat [w x y z]: {np.round(quat, 5)}")
        print(f"gyro      [x y z]:   {np.round(gyro, 5)}")
        print(f"accel     [x y z]:   {np.round(accel, 5)}")
        print(f"upvector  [x y z]:   {np.round(up, 5)}")
        print(f"forward   [x y z]:   {np.round(forward, 5)}")

    def key_callback(self, keycode: int):
        if keycode == 72:
            self._print_help()
        elif keycode == 82:
            self._reset_pose()
        elif keycode == 70:
            self.freeze_joints = not self.freeze_joints
            state = "on" if self.freeze_joints else "off"
            print(f"freeze joints: {state}")
        elif keycode == 55:
            self._rotate_base_local(np.array([1.0, 0.0, 0.0]), -1.0, "roll")
        elif keycode == 56:
            self._rotate_base_local(np.array([1.0, 0.0, 0.0]), 1.0, "roll")
        elif keycode == 57:
            self._rotate_base_local(np.array([0.0, 1.0, 0.0]), -1.0, "pitch")
        elif keycode == 48:
            self._rotate_base_local(np.array([0.0, 1.0, 0.0]), 1.0, "pitch")
        elif keycode == 45:
            self._rotate_base_local(np.array([0.0, 0.0, 1.0]), -1.0, "yaw")
        elif keycode == 61:
            self._rotate_base_local(np.array([0.0, 0.0, 1.0]), 1.0, "yaw")
        elif keycode == 90:
            self._pulse_gyro(0, -1.0, "x")
        elif keycode == 88:
            self._pulse_gyro(0, 1.0, "x")
        elif keycode == 67:
            self._pulse_gyro(1, -1.0, "y")
        elif keycode == 86:
            self._pulse_gyro(1, 1.0, "y")
        elif keycode == 66:
            self._pulse_gyro(2, -1.0, "z")
        elif keycode == 78:
            self._pulse_gyro(2, 1.0, "z")

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()
                if self.freeze_joints:
                    self.data.ctrl[:] = self.home_ctrl
                mujoco.mj_step(self.model, self.data)
                self.counter += 1
                if self.counter % self.print_every == 0:
                    gyro = self._get_sensor_vec(self.gyro_adr)
                    accel = self._get_sensor_vec(self.accel_adr)
                    print(
                        "live gyro [x y z]:",
                        np.round(gyro, 4),
                        " accel [x y z]:",
                        np.round(accel, 4),
                    )
                viewer.sync()
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(
        description="Probe IMU axis order and sign in MuJoCo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/dog_m/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--angle_deg",
        type=float,
        default=10.0,
        help="Angle applied by each pose test key press.",
    )
    parser.add_argument(
        "--pulse_rad_s",
        type=float,
        default=1.0,
        help="Angular velocity pulse used for gyro sign tests.",
    )
    args = parser.parse_args()

    probe = ImuAxisProbe(
        resolve_input_path(args.model_path),
        angle_deg=args.angle_deg,
        pulse_rad_s=args.pulse_rad_s,
    )
    probe.run()


if __name__ == "__main__":
    main()
