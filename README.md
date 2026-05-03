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

## Quick start Train and export ONNX

```bash
python playground/dog_m/runner.py 
```

### Run the exported policy in MuJoCo

```bash
python playground/dog_m/mujoco_infer.py \
  -o outputs/<your_policy>.onnx 
```

### Run the bundled ONNX directly

```bash
python playground/dog_m/mujoco_infer.py \
  -o models/best.onnx 
```

Use this command if you want to verify the repository immediately without training first.

Keyboard controls in the inference viewer:

- Up/Down: forward/backward command
- Left/Right: lateral command
- `A` / `E`: yaw command

## Demo videos

<div align="center">

  <video src="https://github.com/user-attachments/assets/cafc08f6-3ca6-4018-8375-37cb2e9bd252" 
         controls 
         width="80%">
  </video>

  <p><b>Full-stack Embodied AI Quadruped Robot Demo</b></p>
  <p>
    A low-cost open-source quadruped robot pipeline covering mechanical design,
    simulation, reinforcement learning, IK control, and deployment.
  </p>

</div>


### IMU axis probe

```bash
python playground/dog_m/imu_axis_probe.py \
  --model_path playground/dog_m/xmls/scene_flat_terrain.xml
```

This tool helps verify roll, pitch, yaw, gyro, and accelerometer conventions.

