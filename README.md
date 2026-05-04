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


### 1. Recreate the exact `dog_m` conda environment

```bash
cd /root/dog_sim/Dog_Playground
conda env create -f environment.yml
conda activate dog_m
```
That's a GPU environment

If `dog_m` already exists, update it in place:

```bash
conda env update -n dog_m -f environment.yml --prune
conda activate dog_m
```

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

