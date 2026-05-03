from pathlib import Path
import sys

import mujoco
import pickle
import numpy as np
import mujoco.viewer
import time
import argparse

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from playground.common.onnx_infer import OnnxInfer
from playground.common.utils import LowPassActionFilter

from playground.dog_m.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True
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


class MjInfer(MJInferBase):
    def __init__(self, model_path: str, onnx_model_path: str):
        super().__init__(model_path)

        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = np.zeros(3, dtype=np.float32)
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
            ]
        ).astype(np.float32)

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/dog_m/xmls/scene_flat_terrain.xml",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(
        resolve_input_path(args.model_path),
        resolve_input_path(args.onnx_model_path),
    )
    mjinfer.run()
