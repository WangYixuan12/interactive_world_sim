from pathlib import Path


def test_stage1_job_uses_readme_settings_and_two_views() -> None:
    script = Path("jobs/train_iws_libero90_stage1.sbatch").read_text()

    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --partition=cscc-gpu-p" in script
    assert "#SBATCH --qos=cscc-gpu-qos" in script
    assert "dataset=libero90_dataset" in script
    assert 'dataset.obs_keys=[agentview_rgb,eye_in_hand_rgb]' in script
    assert "experiment.training.max_steps=1000005" in script
    assert "experiment.training.batch_size=1" in script
    assert "experiment.validation.val_every_n_step=6000" in script
    assert "experiment.validation.batch_size=10" in script
    assert "experiment.training.checkpointing.every_n_train_steps=10000" in script
    assert "algorithm.latent_dim=512" in script
    assert "algorithm.action_dim=7" in script
    assert "algorithm.training_stage=1" in script
    assert "RESUME_CKPT" in script
    assert 'WANDB_ENTITY="${WANDB_ENTITY:-youssef-ghallab-mbzuai}"' in script
    assert 'IWS_ROOT="${IWS_ROOT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/interactive_world_sim}"' in script
    assert "set +u\nsource /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh\nconda activate /home/youssef.ghallab/miniforge3/envs/iws\nset -u" in script
