# Dog Playground

MuJoCo + JAX based dog locomotion playground for training, ONNX export, and local inference.

## What is included

- `playground/dog_m/runner.py`: training entrypoint
- `playground/dog_m/joystick.py`: locomotion task
- `playground/dog_m/mujoco_infer.py`: run an exported ONNX policy in MuJoCo
- `playground/dog_m/imu_axis_probe.py`: inspect IMU axis conventions in the simulator
- `playground/common/`: shared training, export, reward, and utility code
- `playground/dog_m/xmls/`: robot and scene assets
- `models/best.onnx`: bundled ready-to-run ONNX policy
- `videos/`: reserved folder for demo videos

## Current scope

- Supported task: `flat_terrain`
- Training exports the latest ONNX policy into `outputs/`
- A reference ONNX policy is included at `models/best.onnx`
- This repository no longer keeps training-state snapshots in-tree

## Reference environment

This repository is now aligned to the local conda environment `dog_m` on this machine.

- Conda env name: `dog_m`
- Python: `3.12.13`
- JAX: `0.9.2`
- `jaxlib`: `0.9.2`
- `jax-cuda12-pjrt`: `0.9.2`
- `jax-cuda12-plugin`: `0.9.2`
- MuJoCo: `3.6.0`
- `mujoco-mjx`: `3.6.0`
- `playground` (provides `mujoco_playground`): `0.0.5`
- TensorFlow: `2.21.0`
- `tf2onnx`: `1.17.0`

## Setup

### 1. Recreate the exact `dog_m` conda environment

```bash
cd /root/dog_sim/Dog_Playground
conda env create -f environment.yml
conda activate dog_m
```

If `dog_m` already exists, update it in place:

```bash
conda env update -n dog_m -f environment.yml --prune
conda activate dog_m
```

### 2. Alternative: install the pinned packages into an existing env

```bash
conda create -n dog_m python=3.12.13
conda activate dog_m
pip install -r requirements.txt
```

### 3. Platform note

The current `dog_m` environment is a CUDA 12 GPU environment because it uses:

- `jax-cuda12-pjrt==0.9.2`
- `jax-cuda12-plugin==0.9.2`

If you want a CPU-only environment, replace those two packages with a CPU-compatible JAX install before running the project.

## Quick start

### Show training CLI help

```bash
python playground/dog_m/runner.py --help
```

### Train and export ONNX

```bash
python playground/dog_m/runner.py \
  --task flat_terrain \
  --output_dir outputs \
  --num_timesteps 30000000
```

Notes:

- ONNX export is written to `outputs/`
- only the latest timestamped ONNX file is kept by the runner

### Run the exported policy in MuJoCo

```bash
python playground/dog_m/mujoco_infer.py \
  -o outputs/<your_policy>.onnx \
  --model_path playground/dog_m/xmls/scene_flat_terrain.xml
```

### Run the bundled ONNX directly

```bash
python playground/dog_m/mujoco_infer.py \
  -o models/best.onnx \
  --model_path playground/dog_m/xmls/scene_flat_terrain.xml
```


Use this command if you want to verify the repository immediately without training first.

Keyboard controls in the inference viewer:

- Up/Down: forward/backward command
- Left/Right: lateral command
- `A` / `E`: yaw command

## Demo videos
#######

### IMU axis probe

```bash
python playground/dog_m/imu_axis_probe.py \
  --model_path playground/dog_m/xmls/scene_flat_terrain.xml
```

This tool helps verify roll, pitch, yaw, gyro, and accelerometer conventions.



## Repository layout

```text
Dog_Playground/
├── LICENSE
├── models/
│   └── best.onnx
├── pyproject.toml
├── playground/
│   ├── common/
│   └── dog_m/
│       ├── base.py
│       ├── constants.py
│       ├── imu_axis_probe.py
│       ├── joystick.py
│       ├── mujoco_infer.py
│       ├── mujoco_infer_base.py
│       ├── runner.py
│       └── xmls/
├── videos/
│   └── README.md
├── environment.yml
├── requirements.txt
└── README.md
```

## Common issues

### `ModuleNotFoundError: No module named 'jax'`

Install the pinned `dog_m` environment first, then reinstall the remaining dependencies:

```bash
conda env create -f environment.yml
conda activate dog_m
```

### `ModuleNotFoundError: No module named 'mujoco'`

```bash
pip install mujoco
```

### Viewer does not open

- Make sure you are on a machine with a working desktop/OpenGL setup
- If you are on a remote machine, verify X11 forwarding or your display configuration

## Notes for open-source cleanup

- generated caches and exports should stay out of version control
- `.gitignore` already ignores `outputs/`, `.tmp/`, `__pycache__/`, TensorBoard logs, and temporary observation dumps
- the repository is released under the MIT license in `LICENSE`
