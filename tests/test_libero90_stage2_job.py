import os
from pathlib import Path
import subprocess

from hydra.core.override_parser.overrides_parser import OverridesParser


def test_stage2_job_uses_temporal_dynamics_settings() -> None:
    path = Path("jobs/train_iws_libero90_stage2.sbatch")
    assert path.is_file()
    script = path.read_text()

    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "dataset.horizon=10 dataset.val_horizon=200" in script
    assert "experiment.training.batch_size=$TRAIN_BATCH_SIZE" in script
    assert "experiment.training.max_steps=$MAX_STEPS" in script
    assert "experiment.validation.limit_batch=1" in script
    assert "experiment.validation.batch_size=1" in script
    assert "experiment.validation.val_every_n_step=$VAL_EVERY_N_STEP" in script
    assert (
        "experiment.training.checkpointing.every_n_train_steps=$CHECKPOINT_EVERY_N_STEPS"
        in script
    )
    assert "algorithm.noise_scheduler.loss_weighting=uniform" in script
    assert "algorithm.sampling_strategy=terminal_only" in script
    assert "algorithm.val_render=false" in script
    assert "algorithm.training_stage=2" in script
    assert '"algorithm.load_ae=\'$STAGE1_CKPT\'"' in script
    assert 'MAX_STEPS="${MAX_STEPS:-1000005}"' in script
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"' in script
    assert 'VAL_EVERY_N_STEP="${VAL_EVERY_N_STEP:-30000}"' in script
    assert 'CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"' in script
    assert 'RESUME_CKPT="${RESUME_CKPT:-}"' in script
    assert '[[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]' in script
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in script


def test_stage2_job_rejects_missing_stage1_checkpoint_before_conda(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", "jobs/train_iws_libero90_stage2.sbatch"],
        env={
            **os.environ,
            "STAGE1_CKPT": str(tmp_path / "missing.ckpt"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "STAGE1_CKPT must be a regular file" in result.stderr


def test_stage2_job_quotes_checkpoint_overrides_for_hydra() -> None:
    script = Path("jobs/train_iws_libero90_stage2.sbatch").read_text()
    checkpoint = "/tmp/epoch=0-step=415000.ckpt"

    for key, source in (
        ("algorithm.load_ae", '"algorithm.load_ae=\'$STAGE1_CKPT\'"'),
        ("load", '"load=\'$RESUME_CKPT\'"'),
    ):
        assert source in script
        override = OverridesParser.create().parse_overrides([f"{key}='{checkpoint}'"])[0]
        assert override.value() == checkpoint
