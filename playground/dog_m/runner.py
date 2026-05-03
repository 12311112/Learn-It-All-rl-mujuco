"""Runs training and evaluation loop for Dog M."""

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.dog_m import constants
from playground.dog_m import joystick


class DogMRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick)
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dog M runner script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Where to save exported artifacts",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=30000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument(
        "--task",
        type=str,
        default="flat_terrain",
        choices=constants.AVAILABLE_TASKS,
        help="Task to run",
    )
    args = parser.parse_args()

    runner = DogMRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
