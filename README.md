# Interactive World Simulator for Robot Policy Training and Evaluation

[Yixuan Wang](https://yixuanwang.me/)<sup>1</sup>, [Rhythm Syed](https://rhythmsyed.github.io/)<sup>1</sup>, [Fangyu Wu](https://fangyuwu.com/)<sup>1</sup>, [Mengchao Zhang](https://zmccmzty.github.io/)<sup>2</sup>, [Aykut Onol](https://scholar.google.com/citations?user=mMYFmOYAAAAJ&hl=en)<sup>2</sup>, [Jose Barreiros](https://scholar.google.com/citations?user=mLFRRpkAAAAJ&hl=en)<sup>2</sup>, [Hooshang Nayyeri](https://www.amazon.science/author/hooshang-nayyeri)<sup>3</sup>, [Tony Dear](https://tonydear.com/)<sup>1</sup>, [Huan Zhang](https://www.huan-zhang.com/)<sup>4</sup>, [Yunzhu Li](https://yunzhuli.github.io/)<sup>1</sup>

<sup>1</sup>Columbia University &emsp; <sup>2</sup>Toyota Research Institute &emsp; <sup>3</sup>Amazon &emsp; <sup>4</sup>University of Illinois Urbana-Champaign

**[Paper](https://www.yixuanwang.me/interactive_world_sim/texts/main.pdf) | [Project Page](https://www.yixuanwang.me/interactive_world_sim/) | [Video](https://youtu.be/H6Um4zZYm5Y) | [Code](https://github.com/WangYixuan12/interactive_world_sim)**



https://github.com/user-attachments/assets/78d04003-4b1e-4844-8115-3a2a05753723



## Table of Contents
- [🔨 Installation](#-installation)
- [🤖 Inference](#-inference)
  - [Download Checkpoints](#download-checkpoints-a-iddownload-ckptsa)
  - [Download Data](#download-data-a-iddownload-dataa)
  - [Teleoperate from Keyboard](#teleoperate-from-keyboard)
  - [Teleoperate from ALOHA Robot](#teleoperate-from-aloha-robot-a-idaloha-teleopa)
- [🖥️ Local Interactive Demo](#️-local-interactive-demo)
- [🏋️ Training](#️-training)
  - [Stage 1: Autoencoder Training](#stage-1-autoencoder-training)
  - [Stage 2: Dynamics Training](#stage-2-dynamics-training)
  - [Stage 3: Autoencoder Finetuning](#stage-3-autoencoder-finetuning)
- [📦 Real-World Data Collection on ALOHA](#-real-world-data-collection-on-aloha)
- [🌎 WM Data Collection on ALOHA](#-wm-data-collection-on-aloha)

## 🔨 Installation

**Step 1**: Create and activate the conda environment.
```bash
mamba env create -f conda_env.yaml
conda activate iws
```

**Step 2**: Install Python dependencies.
```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
```

**Step 3**: Install the package in editable mode.
```bash
pip install -e .
```

**Step 4**: Configure your [Weights & Biases](https://wandb.ai/) entity in `configurations/config.yaml`:
```yaml
wandb:
  entity: YOUR_WANDB_ENTITY
```

## 🤖 Inference

### Download Checkpoints <a id="download-ckpts"></a>

Download all pretrained checkpoints with one command (requires `gdown`, installed automatically):

```bash
bash scripts/download_checkpoints.sh
```

This downloads 7 checkpoints to `outputs/`, each with its Hydra config:

| Directory | Task | Camera(s) |
|-----------|------|-----------|
| `outputs/pusht_cam1/` | PushT | cam1 |
| `outputs/single_grasp_cam0/` | Single Grasp | cam0 |
| `outputs/single_grasp_cam1/` | Single Grasp | cam1 |
| `outputs/bimanual_sweep_cam0/` | Bimanual Sweep | cam0 |
| `outputs/bimanual_sweep_cam1/` | Bimanual Sweep | cam1 |
| `outputs/bimanual_rope_cam0/` | Bimanual Rope | cam0 |
| `outputs/bimanual_rope_cam1/` | Bimanual Rope | cam1 |

Each directory contains `checkpoints/best.ckpt` and `.hydra/config.yaml`.

Alternatively, download the full [Google Drive folder](https://drive.google.com/drive/folders/1wVvQMcGjjyCLH7aeqpeQbDsdcBWximSA?usp=sharing) manually.

### Download Data <a id="download-data"></a>

**Mini dataset** (for inference and debugging — a few episodes per task):

```bash
bash scripts/download_mini_data.sh
```

Downloads to `data/mini/{pusht,single_grasp,bimanual_sweep,bimanual_rope}/`. ([Google Drive](https://drive.google.com/drive/folders/1zTWiVq1SsjuxCBIt6qtlITC9Hp9a4xT1?usp=drive_link))

**Full training dataset**:

```bash
bash scripts/download_full_data.sh
```

Downloads to `data/full/{pusht,single_grasp,bimanual_sweep,bimanual_rope}/`. ([Google Drive](https://drive.google.com/drive/folders/1umtKJI5TmIGxiD3TecjSwLqc9Lik5Gls?usp=drive_link))

### Teleoperate from Keyboard

Use the keyboard to teleoperate the robot through the world model (no physical robot required). Example for the PushT task:

```bash
python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' \
  +use_joystick=false \
  +use_dataset=false \
  +act_horizon=1 \
  +scene=real \
  "+ckpt_paths=['outputs/pusht_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht/val \
  "dataset.obs_keys=['camera_1_color']"
```

**Controls:** WASD to move left end-effector; IKJL to move right end-effector; Press "c" to start recording into HDF5 file; Press "s" to stop recording and save into HDF5 file; Press "q" to abandon HDF5 file recording.

Inference scripts for all camera views and tasks are in `scripts/inference/keyboard/`. Controls vary by task:

| Task | Script(s) | Keys | Action |
|------|-----------|------|--------|
| PushT | `pusht_kybd.sh` | WASD | Move left arm XY |
| | | IJKL | Move right arm XY |
| Single Grasp | `single_grasp_cam{0,1}_kybd.sh` | WASD | Move arm XY |
| | | IK | Move arm Z |
| | | JL | Open / close gripper |
| Bimanual Sweep | `bimanual_sweep_cam{0,1}_kybd.sh` | WASD | Move left arm XY |
| | | IJKL | Move right arm XY |
| Bimanual Rope | `bimanual_rope_cam{0,1}_kybd.sh` | WASD | Move left arm XY |
| | | IJKL | Move right arm XY |
| | | QE | Move left arm Z |
| | | UO | Move right arm Z |

### Teleoperate from ALOHA Robot <a id="aloha-teleop"></a>

First, follow ALOHA setup process [here](https://github.com/WangYixuan12/gendp?tab=readme-ov-file#set-up-robot) to set up real robots. Example for Single Grasp with both cameras:

```bash
python scripts/inference/teleoperate_aloha.py \
  +output_dir='data/wm_demo' \
  +act_horizon=1 \
  +scene=single_grasp_cam_0 \
  "+ckpt_paths=['outputs/single_grasp_cam0/checkpoints/best.ckpt', 'outputs/single_grasp_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/real_aloha/single_grasp/val \
  "dataset.obs_keys=['camera_0_color', 'camera_1_color']"
```

All scripts are in `scripts/inference/aloha/`. Each task has three variants:

| Task | Script(s) | Cameras |
|------|-----------|---------|
| Single Grasp | `single_grasp_cam0_aloha.sh` | cam0 only |
| | `single_grasp_cam1_aloha.sh` | cam1 only |
| | `single_grasp_cam0_and_cam1_aloha.sh` | both |
| Bimanual Sweep | `bimanual_sweep_cam0_aloha.sh` | cam0 only |
| | `bimanual_sweep_cam1_aloha.sh` | cam1 only |
| | `bimanual_sweep_cam0_and_cam1_aloha.sh` | both |
| Bimanual Rope | `bimanual_rope_cam0_aloha.sh` | cam0 only |
| | `bimanual_rope_cam1_aloha.sh` | cam1 only |
| | `bimanual_rope_cam0_and_cam1_aloha.sh` | both |

## 🖥️ Local Interactive Demo

Interact with the world model live in your browser!

1. Start the server: `bash deploy/start_demo.sh`

2. Open [https://www.yixuanwang.me/interactive_world_sim/](https://www.yixuanwang.me/interactive_world_sim/) in your browser and click **Connect Locally**

> **Requirements:** finish [Download Checkpoints](#download-ckpts) and [Download Data](#download-data) (mini dataset).



## 🏋️ Training

Training uses [Weights & Biases](https://wandb.ai/) for logging. Make sure your entity is configured in `configurations/config.yaml`. Here we show example scripts to train the world model for `T Pushing` task.

### Stage 1: Autoencoder Training

Train the encoder and diffusion decoder to compress RGB observations into a compact latent space.

```bash
python main.py +name=pusht_stage_1 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  experiment.training.batch_size=1 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1
```

My stage 1 training report is attached [here](https://api.wandb.ai/links/yixuan1999/a36tjf2d) for reference.

### Stage 2: Dynamics Training

Train the latent dynamics model to predict future latent states from past observations and actions. Requires a Stage 1 checkpoint.

```bash
python main.py +name=pusht_stage_2 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht \
  dataset.horizon=10 dataset.val_horizon=200 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  experiment.training.batch_size=4 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="path_to_stage_1.ckpt" \
  algorithm.training_stage=2
```

My stage 2 training report is attached [here](https://api.wandb.ai/links/yixuan1999/kl8zupkr) for reference.

### Stage 3: Autoencoder Finetuning

Finetune the decoder to make it robust to latent noises.

```bash
python main.py +name=pusht_stage_3 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht \
  dataset.horizon=1 dataset.val_horizon=200 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="path_to_stage_2.ckpt" \
  algorithm.training_stage=3
```

My stage 3 training report is attached [here](https://api.wandb.ai/links/yixuan1999/zqw1p9rc) for reference.

### Empirical tips for training:
- For stage 1 training, the reconstruction should be **almost perfect** before proceeding to stage 2.
- For new tasks, you just need to change `action_mode` and `action_dim` accordingly.
- To train a good model, the play dataset is suggested to have large action data coverage (different speed, contact modes, and positions) and minimal occlusion.
- Order of stage 2 and stage 3 can be swapped.
- Even after the validation metrics converge, you are suggested to wait for longer to achieve the best result.
- Stage 2 training is most time-consuming. Stage 1 takes less time than stage 2 but more time than stage 3.

## 📦 Real-World Data Collection on ALOHA

First, follow ALOHA setup process [here](https://github.com/WangYixuan12/gendp?tab=readme-ov-file#set-up-robot) to set up real robots. Example commands of recording HDF5 episodes are shown below.

```bash
python scripts/data_collection/collect_real_aloha.py \
  --output_dir data/bimanual_push \
  --robot_sides right \
  --robot_sides left \
  --frequency 10 \
  --ctrl_mode bimanual_push \
  --total_steps 200
```

You need to change `ctrl_mode` for different tasks. After the data collection, you could run the following command to sleep robots safely:

```bash
python -m interactive_world_sim.real_world.robot_sleep --left --right
```

**Requirements:**
- ALOHA robot hardware (see [`real_world/`](real_world/))
- Intel RealSense cameras (configured in `real_world/aloha_extrinsics/`)

Data is saved in HDF5 format and cached as a zarr dataset for fast loading during training.


## 🌎 WM Data Collection on ALOHA

You could reuse commands from [Teleoperate from ALOHA Robot](#aloha-teleop) to collect data.
Press "c" to start recording into HDF5 file; Press "s" to stop recording and save into HDF5 file; Press "q" to abandon HDF5 file recording.


## Acknowledgements

- Built on [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing).

> This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.
